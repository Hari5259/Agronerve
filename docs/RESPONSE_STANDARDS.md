# AgroNerve Agricultural Response Standards & Architecture 🌾

## 1. Overview
AgroNerve adheres to strict agronomic response quality, regulatory safety standards (CIBRC / ICAR / FAO-56), and structured formatting for clear farmer communication.

---

## 2. Response Structure

Every response synthesized by AgroNerve adheres to the following layout:

```markdown
`🔬 Crop Pathology Advisory`  `🌱 Target Crop: Tomato`

> ⚠️ **CRITICAL SAFETY ADVISORY:** (If chemical sprays or toxic hazards exist)

### 📍 Diagnosis / Field Action
[Concise root cause, causal organism, or meteorological impact]

### 🌿 Organic & Biological Solutions
- Eco-friendly bio-pesticides (e.g. Neem Oil 10,000 ppm, Beauveria, Trichoderma)

### 🛡️ Verified Chemical Controls & Dosages
| Chemical / Bio-Agent | Dilution Rate | Application Method | PHI (Days) |
|---|---|---|---|
| Coragen 18.5 SC | 0.4 ml/L | Foliar Spray | 14 days |

### 💧 Soil & Irrigation Protocol
- Soil moisture adaptations, watering intervals, and rain-fastness hours.
```

---

## 3. Safety Guardrails & CIBRC Compliance

AgroNerve runs all generated advisories through `core.safety_guardrails.SafetyGuardrails`:

1. **Banned Chemical Interception:**
   - Detects banned substances (e.g., Endosulfan, Monocrotophos on vegetables, Methyl Parathion, Paraquat misuse) and injects regulatory warnings.
2. **Mandatory PPE Enforcement:**
   - Injects personal protective equipment (nitrile gloves, mask, eye protection) for any chemical recommendation.
3. **Pre-Harvest Interval (PHI) Verification:**
   - Mandates explicit waiting intervals before produce picking to prevent pesticide residue poisoning.
4. **Concentration Guardrails:**
   - Flags concentrations exceeding recommended safe limits.

---

## 4. Precision Agronomic Calculators

Located in `core/agri_calculator.py`:

- **Spray Dilution & Refills:** Calculates exact total water volume, chemical quantity, and knapsack tank refills.
- **Crop Water Requirement (ETc):** Uses FAO-56 crop coefficients ($K_c$) and reference evapotranspiration ($ET_o$).
- **Fertilizer NPK Split:** Translates N-P-K nutrient targets into commercial bags of Urea, DAP, and MOP.

---

## 5. Multilingual Localization

Full dictionary support in English, Tamil, Hindi, Telugu, and Kannada with spoken voice TTS phonetics in `core/voice_engine.py`.
