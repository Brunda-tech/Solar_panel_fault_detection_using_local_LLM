def rule_based_diagnosis(V, I, P, T):
    
    # Bypass Diode Failure first — highest priority (sudden crash)
    if V < 5 and I < 0.05 and P < 0.3:
        return "Bypass Diode Failure"

    # Hotspot — V and I drop AND temp spikes
    if V < 10 and I < 0.12 and T >= 70:
        return "Hotspot"

    # Partial Shading — both V and I drop, temp normal
    if V < 10 and I < 0.12 and T < 70:
        return "Partial Shading"

    # Soiling — current drops, voltage fine, temp normal
    if I < 0.09 and V >= 10 and T < 70:
        return "Soiling"

    return "Normal"

# Test it
test_cases = [
    {"V": 11.5, "I": 0.18, "P": 2.07, "T": 40},   # Normal
    {"V": 11.2, "I": 0.08, "P": 0.90, "T": 42},   # Soiling
    {"V": 8.5,  "I": 0.10, "P": 0.85, "T": 38},   # Partial Shading
    {"V": 8.0,  "I": 0.09, "P": 0.72, "T": 85},   # Hotspot
    {"V": 3.0,  "I": 0.02, "P": 0.06, "T": 45},   # Bypass Diode Failure
]

for t in test_cases:
    result = rule_based_diagnosis(t["V"], t["I"], t["P"], t["T"])
    print(f"V={t['V']} I={t['I']} T={t['T']} → {result}")
