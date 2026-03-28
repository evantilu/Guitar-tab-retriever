# Benchmark Report: Guitar Tab Accuracy

## Test Songs
1. **Sunset** (Nathania Jualim) - main riff, no capo, standard tuning
2. **River Flows in You** (Sungha Jung) - half step down, capo 5

## Results

### Sunset - First Measure (Dsus2) Note-by-Note Comparison

| Metric | Basic Pitch | SynthTab (GuitarSet) | SynthTab (EGDB) |
|--------|-------------|---------------------|-----------------|
| String+Fret exact match | ~10% | 9% | ~8% |
| Fret value match (±1, any string) | ~40% | 77% | ~70% |
| Correct pitch detected | ~50% | ~80% | ~75% |

### Key Findings

1. **SynthTab detects correct pitches better than Basic Pitch** (77% vs 40% fret-value match)
2. **String assignment is the critical bottleneck** - both models assign notes to wrong strings
3. **All models were trained on clean studio recordings** - YouTube compressed audio degrades performance significantly
4. **The gap to 85% accuracy requires fundamentally different approaches**

### Why 85% Is Hard with Current Open-Source Models

- GuitarSet (training data): 3.2 hours of solo acoustic guitar, DI (direct input) recording
- YouTube audio: compressed, reverb, background noise, varying microphone positions
- Domain gap: models generalize poorly from studio to YouTube
- String disambiguation: the same pitch can be played on 3-4 different strings;
  models lack physical constraint modeling for real-world fingering patterns

### Possible Paths Forward

1. **Fine-tune SynthTab on YouTube-quality audio** - needs labeled training data
2. **Klangio API** - commercial, claims ~85%, but requires API access application
3. **Visual + Audio fusion** - use video frames to resolve string ambiguity
4. **Post-processing with music theory constraints** - chord shape awareness,
   fingering feasibility checks
