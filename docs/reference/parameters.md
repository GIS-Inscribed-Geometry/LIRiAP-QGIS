# Parameter Reference

Complete parameter reference for all LIRiAP algorithms.

## Common Parameters

### Input/Output
| Parameter | Type | Description | Applies To |
|-----------|------|-------------|------------|
| INPUT | VectorLayer | Input polygon layer | All |
| OUTPUT | FeatureSink | Output rectangle layer | All |

### Search Control
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| GRID_COARSE | Integer | 40 | Initial grid resolution for coarse search (Approximation, Contained, BCRS) |
| GRID_FINE | Integer | 100/120 | Fine grid resolution for refinement |
| ANGLE_STEP | Double | 5.0 | Fallback sweep angle step in degrees |
| TOP_K | Integer | 3 | Number of angle candidates to keep for refinement (Contained, BCRS) |

### Constraint
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| MAX_RATIO | Double | 1.6 | Maximum aspect ratio (long:short), 0 = unlimited |

### Containment
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ALWAYS_RETURN | Boolean | True | Return best-effort rectangle if strict certification fails |
| USE_BUFFER | Boolean | False | Apply containment buffer after certification |
| BUFFER_VALUE | Double | -0.5 | Buffer distance in map units (negative = inward) |

### Performance
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| N_WORKERS | Integer | 1 | Workers (0 = auto, 1 = serial, >1 = parallel) |
| USE_CHUNKING | Boolean | False | Enable chunked parallel execution |
| AUTO_INSTALL_NUMBA | Boolean | False | Attempt Numba auto-install if missing |

### Axis-Aligned Specific
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| AXIS_ANGLE | Double | 0.0 | Rotation of axis-aligned frame in degrees |

## Algorithm-Specific Defaults

| Algorithm | GRID_COARSE | GRID_FINE | TOP_K | ANGLE_STEP |
|-----------|-------------|-----------|-------|------------|
| Approx Standard | 40 | 100 | 1 | 5.0 |
| Approx Fast | 40 | 100 | 1 | 5.0 |
| Contained Standard | 40 | 120 | 3 | 5.0 |
| Contained Fast | 40 | 120 | 3 | 5.0 |
| BCRS | 40 | 120 | 3 | 5.0 |
| BCRS Fast | 40 | 120 | 3 | 5.0 |
| Axis-Aligned | N/A | 120 (fallback) | N/A | N/A |

## Parameter Effects

### GRID_COARSE / GRID_FINE
- Higher values → more accurate results, slower execution
- Trade-off: quadratic time complexity in grid size

### ANGLE_STEP
- Smaller values → more angle candidates, slower but potentially more accurate
- Affects fallback sweep only (primary candidates from edge directions)

### TOP_K
- Higher values → more candidates refined, slower but potentially better
- Must be >= 1

### MAX_RATIO
- Lower values → more constrained shapes, may reduce max area
- 0 = no limit

### ALWAYS_RETURN
- True: Always return a rectangle (may be smaller if best-effort used)
- False: Return nothing if strict certification fails

### N_WORKERS
- 1: Serial execution
- >1: Parallel execution (per-feature or chunked)
- 0: Auto-detect CPU count

### USE_CHUNKING
- True: Chunk-based parallel (better for canceling)
- False: Per-feature parallel

## Output Fields

### Approximation
| Field | Type | Description |
|-------|------|-------------|
| feat_id | int | Source feature ID |
| area | double | Rectangle area |
| angle | double | Rotation angle (degrees) |
| ratio | double | Aspect ratio |

### Contained / BCRS
| Field | Type | Description |
|-------|------|-------------|
| feat_id | int | Source feature ID |
| area | double | Rectangle area |
| angle | double | Rotation angle (degrees) |
| ratio | double | Aspect ratio |
| cand_rank | int | Rank of selected candidate |
| s2_gain | double | Area gain from Stage 2 |
| s4_gain | double | Area gain from Stage 4 (BCRS only) |
| s5_gain | double | Area gain from Stage 5 (BCRS only) |
| best_effort | int | 1 if fallback used |

### Axis-Aligned
| Field | Type | Description |
|-------|------|-------------|
| feat_id | int | Source feature ID |
| area | double | Rectangle area |
| axis_angle | double | Axis angle (echoes input) |
| poly_type | string | Polygon type classification |
| ratio | double | Aspect ratio |
| best_effort | int | 1 if fallback used |