# Data Caching System

The pipeline implements intelligent caching to speed up large dataset workflows.

## Cached Files

### 1. Featurized Data (`data/processed/processed.pkl`)
**Created by**: `python -m src.featurize`
**Contains**: Full featurized dataset with fingerprints and descriptors
**Speed improvement**: 10-100x faster loading vs CSV

### 2. Train/Test Split (`data/processed/train_test_cache.pkl`)
**Created by**: `python -m src.train`
**Contains**: Pre-split X_train, X_test, y_train, y_test matrices
**Benefit**: Ensures consistent evaluation across multiple model training runs

### 3. Trained Models (`models/*.pkl`)
**Created by**: `python -m src.train`
**Contains**: Trained sklearn models ready for prediction
**Models**: Ridge, Lasso, ElasticNet, SGD, Decision Tree, KNN, Random Forest

## How It Works

### First Run (1000+ molecules)
```bash
python -m src.data_download --large  # ~30-60 min, creates molecules.csv
python -m src.featurize                # ~10-20 min, creates processed.csv + processed.pkl
python -m src.train                    # ~5-10 min, creates models + train_test_cache.pkl
```

### Subsequent Runs (with cache)
```bash
# If you want to retrain models with different hyperparameters:
python -m src.train                    # ~2-5 min (uses cached pkl files!)

# If you add new molecules:
python -m src.data_download --large    # Re-download
python -m src.featurize                # Re-featurize (updates cache)
python -m src.train                    # Train (uses new cache)
```

## Cache Behavior

### Automatic Cache Usage
The training script automatically:
1. Checks for `processed.pkl` → loads if exists (fast)
2. Falls back to `processed.csv` if pkl missing (slower)
3. Checks for `train_test_cache.pkl` → ensures consistent splits
4. Creates new cache files if missing

### When to Clear Cache

Clear cache when:
- Adding/removing molecules
- Changing feature engineering parameters
- Want fresh train/test split

```bash
# Clear all caches
rm data/processed/processed.pkl
rm data/processed/train_test_cache.pkl

# Clear just the split (re-randomize)
rm data/processed/train_test_cache.pkl
```

## Performance Comparison

### Large Dataset (1000 molecules)

**Without caching:**
- Load CSV: ~30s
- Prepare features: ~5s
- **Total**: ~35s per training run

**With caching:**
- Load pkl: ~0.5s
- Load split cache: ~0.2s
- **Total**: ~0.7s per training run

**Speed improvement**: ~50x faster! ⚡

## Storage Impact

Example sizes for 1000 molecules:
- `molecules.csv`: ~500 KB
- `processed.csv`: ~50 MB (with 2048-bit fingerprints)
- `processed.pkl`: ~50 MB (similar, but faster to load)
- `train_test_cache.pkl`: ~40 MB
- `*.pkl` models: ~1-10 MB each

**Total cache**: ~100-150 MB for 1000 molecules

## Best Practices

1. **Development**: Use small test set (no cache needed)
2. **Experimentation**: Use `--sample 100-200` to test pipeline
3. **Production**: Use `--large` and rely on caching
4. **Model tuning**: Cache makes rapid iteration possible
5. **Clear cache** when changing data or features

## Troubleshooting

**"Loaded CSV" message during training?**
- Cache not created yet. Run featurization first.

**Different results between runs?**
- train_test_cache.pkl ensures consistency
- Delete it if you want a new random split

**Disk space issues?**
- Clear old pkl files: `rm data/processed/*.pkl`
- They'll be recreated when needed
