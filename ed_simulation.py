"""
Emergency Department Simulation using SimPy
============================================
Models a hospital ED with:
- Patient arrivals (Poisson process)
- Triage assessment
- Doctor consultations
- Lab tests (for some patients)
- Bed occupancy
- Shift-based doctor availability

Each simulation run = one configuration of the ED (staffing, arrival rate, etc.)
Each patient within a run = one data record saved to the dataset.

Target variable: did the patient wait > 4 hours? (1 = breach, 0 = no breach)
"""

import simpy
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────
# 1. PATIENT DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class Patient:
    """Holds all recorded data for one patient visit."""
    patient_id: int
    arrival_time: float
    triage_level: int          # 1 (critical) to 5 (minor)
    needs_lab: bool            # whether lab test is required

    # Filled in as the patient moves through the ED
    triage_end_time: float = 0.0
    doctor_start_time: float = 0.0
    doctor_end_time: float = 0.0
    lab_start_time: float = 0.0
    lab_end_time: float = 0.0
    departure_time: float = 0.0

    # Queue state at arrival (snapshot of ED congestion)
    queue_length_on_arrival: int = 0
    num_doctors_on_shift: int = 0

    @property
    def total_time(self) -> float:
        """Total time from arrival to departure in minutes."""
        return self.departure_time - self.arrival_time

    @property
    def wait_for_doctor(self) -> float:
        """Time spent waiting before seeing a doctor."""
        return self.doctor_start_time - self.triage_end_time

    @property
    def breached(self) -> int:
        """Target variable: 1 if total time > 240 minutes (4 hours)."""
        return 1 if self.total_time > 240 else 0


# ─────────────────────────────────────────────
# 2. ED SIMULATION CLASS
# ─────────────────────────────────────────────

class EmergencyDepartment:
    """
    Discrete-event simulation of a hospital Emergency Department.

    Parameters
    ----------
    env               : SimPy environment
    num_doctors       : number of doctors on shift
    num_triage_nurses : number of triage nurses
    num_beds          : number of ED beds available
    arrival_rate      : mean patient arrivals per hour
    lab_capacity      : number of lab processing slots
    """

    def __init__(self, env: simpy.Environment,
                 num_doctors: int,
                 num_triage_nurses: int,
                 num_beds: int,
                 arrival_rate: float,
                 lab_capacity: int):

        self.env = env
        self.arrival_rate = arrival_rate

        # SimPy resources — these are the bottlenecks in the system
        self.doctors = simpy.PriorityResource(env, capacity=num_doctors)
        self.beds = simpy.PriorityResource(env, capacity=num_beds)
        self.triage_nurses = simpy.PriorityResource(env, capacity=num_triage_nurses)
        self.lab = simpy.Resource(env, capacity=lab_capacity)  # no priority here

        self.num_doctors = num_doctors
        self.patients: List[Patient] = []
        self.patient_counter = 0

    def run(self, sim_duration: float = 1440):
        """
        Start the simulation.
        sim_duration: how long to run in minutes (default 480 = 8 hours)
        """
        self.env.process(self.patient_generator(sim_duration))
        self.env.run(until=sim_duration)

    def patient_generator(self, sim_duration: float):
        """
        Generates patients arriving at the ED.
        Inter-arrival times follow an exponential distribution (Poisson arrivals).
        arrival_rate is in patients/hour, so we convert to minutes.
        """
        while True:
            # Time between arrivals in minutes
            inter_arrival = random.expovariate(self.arrival_rate / 60)
            yield self.env.timeout(inter_arrival)

            if self.env.now > sim_duration:
                break

            self.patient_counter += 1
            patient = Patient(
                patient_id=self.patient_counter,
                arrival_time=self.env.now,
                triage_level=self._assign_triage_level(),
                needs_lab=random.random() < 0.65,   # 65% of patients need a lab test
                queue_length_on_arrival=len(self.doctors.queue),
                num_doctors_on_shift=self.num_doctors
            )
            self.patients.append(patient)

            # Start the patient's journey through the ED
            self.env.process(self.patient_journey(patient))

    def _assign_triage_level(self) -> int:
        """
        Assigns triage level based on realistic ED distributions.
        Level 1 = most critical, Level 5 = least urgent.
        Distribution approximates real ED triage data.
        """
        return random.choices(
            population=[1, 2, 3, 4, 5],
            weights=[0.05, 0.15, 0.35, 0.30, 0.15]
        )[0]

    def _triage_duration(self, triage_level: int) -> float:
        """Triage assessment takes 5–15 minutes depending on severity."""
        base = {1: 12, 2: 10, 3: 8, 4: 6, 5: 5}[triage_level]
        return max(1, random.gauss(base, 2))

    def _doctor_consultation_duration(self, triage_level: int) -> float:
        """
        Doctor consultation time varies by triage level.
        Critical patients (level 1) take much longer.
        """
        base = {1: 60, 2: 40, 3: 25, 4: 15, 5: 10}[triage_level]
        return max(5, random.gauss(base, base * 0.2))

    def _lab_duration(self) -> float:
        """Lab tests take 30–90 minutes."""
        return max(20, random.gauss(55, 15))

    def patient_journey(self, patient: Patient):
        """
        Defines the full pathway of a patient through the ED.
        Each 'yield' is a point where the patient waits for a resource.

        Flow:
        Arrival → Triage → Wait for bed → Wait for doctor
               → [Lab test if needed] → Discharge
        """

        # ── STEP 1: TRIAGE ──────────────────────────────────
        with self.triage_nurses.request(priority=patient.triage_level) as req:
            yield req
            yield self.env.timeout(self._triage_duration(patient.triage_level))
            patient.triage_end_time = self.env.now

        # ── STEP 2: WAIT FOR BED + DOCTOR (simultaneously) ──
        # Patient needs both a bed and a doctor
        bed_req = self.beds.request(priority=patient.triage_level)
        doc_req = self.doctors.request(priority=patient.triage_level)

        yield bed_req & doc_req   # wait until BOTH are available

        patient.doctor_start_time = self.env.now

        # Release bed request tracking (bed held until discharge)
        yield self.env.timeout(self._doctor_consultation_duration(patient.triage_level))
        patient.doctor_end_time = self.env.now

        # ── STEP 3: LAB TEST (if required) ──────────────────
        if patient.needs_lab:
            with self.lab.request() as lab_req:
                yield lab_req
                patient.lab_start_time = self.env.now
                yield self.env.timeout(self._lab_duration())
                patient.lab_end_time = self.env.now

        # ── STEP 4: DISCHARGE ────────────────────────────────
        patient.departure_time = self.env.now
        self.doctors.release(doc_req)
        yield self.env.timeout(random.uniform(10, 30))  # discharge delay
        self.beds.release(bed_req)


# ─────────────────────────────────────────────
# 3. SINGLE RUN → DATAFRAME
# ─────────────────────────────────────────────

def run_single_simulation(params: dict, sim_duration: float = 1440) -> pd.DataFrame:
    """
    Runs one simulation with given parameters.
    Returns a DataFrame where each row is one patient.

    Parameters in params dict:
    - num_doctors       : int
    - num_triage_nurses : int
    - num_beds          : int
    - arrival_rate      : float (patients/hour)
    - lab_capacity      : int
    - random_seed       : int
    """
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    env = simpy.Environment()
    ed = EmergencyDepartment(
        env=env,
        num_doctors=params["num_doctors"],
        num_triage_nurses=params["num_triage_nurses"],
        num_beds=params["num_beds"],
        arrival_rate=params["arrival_rate"],
        lab_capacity=params["lab_capacity"]
    )
    ed.run(sim_duration=sim_duration)

    records = []
    for p in ed.patients:
        # Only include patients who fully completed their journey
        if p.departure_time == 0.0:
            continue

        records.append({
            # Simulation config features (what the ML model will use)
            "num_doctors":            params["num_doctors"],
            "num_triage_nurses":      params["num_triage_nurses"],
            "num_beds":               params["num_beds"],
            "arrival_rate":           params["arrival_rate"],
            "lab_capacity":           params["lab_capacity"],

            # Patient-level features
            "triage_level":           p.triage_level,
            "needs_lab":              int(p.needs_lab),
            "arrival_hour":           round((p.arrival_time % 1440) / 60, 2),  # hour within 24hr day
            "queue_length_on_arrival": p.queue_length_on_arrival,

            # Derived time features
            "wait_for_doctor_mins":   round(p.wait_for_doctor, 2),
            "total_time_mins":        round(p.total_time, 2),

            # Target variable
            "breached":               p.breached
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 4. PARAMETER SPACE
# ─────────────────────────────────────────────

# These are the ranges we'll randomly sample from for each simulation run.
# Intentionally includes stressed configurations (few doctors, high arrivals)
# so the dataset contains a meaningful number of breaches (~25-40% target).
PARAM_BOUNDS = {
    "num_doctors":       (2, 6),      # 2 to 6 doctors on shift
    "num_triage_nurses": (1, 4),      # 1 to 4 triage nurses
    "num_beds":          (5, 12),   # 5 to 12 ED beds
    "arrival_rate":      (10.0, 30.0), # 10 to 30 patients per hour (realistic ED range)
    "lab_capacity":      (1, 4),      # 1 to 4 lab processing slots
}

# Stress tiers: we intentionally sample ~40% of runs from "high stress" configs
# This ensures enough breach events for ML to learn from
STRESS_CONFIGS = [
    {"num_doctors": 2, "num_triage_nurses": 1, "num_beds": 8,  "arrival_rate": 12.0, "lab_capacity": 1},
    {"num_doctors": 3, "num_triage_nurses": 2, "num_beds": 10, "arrival_rate": 14.0, "lab_capacity": 2},
    {"num_doctors": 2, "num_triage_nurses": 2, "num_beds": 10, "arrival_rate": 13.0, "lab_capacity": 1},
    {"num_doctors": 3, "num_triage_nurses": 1, "num_beds": 8,  "arrival_rate": 13.0, "lab_capacity": 2},
]


def sample_params(seed: int) -> dict:
    """
    Randomly sample one set of simulation parameters.
    40% of the time: use a pre-defined high-stress config (guarantees breach events).
    60% of the time: fully random sampling across the parameter space.
    """
    rng = random.Random(seed)

    if rng.random() < 0.40:
        # Pick a stress config and add small noise so they're not identical
        base = rng.choice(STRESS_CONFIGS).copy()
        base["num_doctors"]   = max(2, min(5, base["num_doctors"] + rng.randint(-1, 1)))
        base["arrival_rate"]  = round(max(10.0, min(16.0, base["arrival_rate"] + rng.uniform(-1, 1))), 1)
        base["random_seed"]   = seed
        return base

    return {
        "num_doctors":       rng.randint(*PARAM_BOUNDS["num_doctors"]),
        "num_triage_nurses": rng.randint(*PARAM_BOUNDS["num_triage_nurses"]),
        "num_beds":          rng.randint(*PARAM_BOUNDS["num_beds"]),
        "arrival_rate":      round(rng.uniform(*PARAM_BOUNDS["arrival_rate"]), 1),
        "lab_capacity":      rng.randint(*PARAM_BOUNDS["lab_capacity"]),
        "random_seed":       seed
    }


# ─────────────────────────────────────────────
# 5. QUICK TEST — run 3 simulations
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running 10 test simulations...\n")

    all_data = []
    for i in range(10):
        params = sample_params(seed=i)
        df = run_single_simulation(params)
        all_data.append(df)
        breach = df["breached"].mean()
        stress = "⚠ stressed" if params["arrival_rate"] >= 17 or params["num_doctors"] <= 3 else "  normal"
        print(f"Sim {i+1:2d} [{stress}] | doctors={params['num_doctors']} "
              f"arrival={params['arrival_rate']}/hr | "
              f"{len(df):3d} patients | breach={breach:.1%}")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records : {len(combined)}")
    print(f"Overall breach rate : {combined['breached'].mean():.1%}")
    print(f"Breach counts : {combined['breached'].sum()} breached / "
          f"{(combined['breached']==0).sum()} not breached")

    # Target: breach rate between 20% and 45%
    rate = combined["breached"].mean()
    if rate < 0.20:
        print("\n⚠  Breach rate too low — increase arrival_rate or decrease num_doctors bounds")
    elif rate > 0.50:
        print("\n⚠  Breach rate too high — decrease arrival_rate or increase num_doctors bounds")
    else:
        print(f"\n✓  Breach rate looks good for ML training ({rate:.1%})")

    print(f"\nFeature columns: {list(combined.columns)}")
    combined.to_csv("test_simulation_output.csv", index=False)
    print("\nSaved to test_simulation_output.csv")
