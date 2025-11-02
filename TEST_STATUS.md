# Test Status

## Current State

**Cannot run tests yet** - Missing system dependencies.

## Required Dependencies

```bash
sudo apt install libclang-dev libpam0g-dev
```

Or run:
```bash
./INSTALL_DEPS.sh
```

## Once Dependencies Are Installed

### Run Unit Tests
```bash
cargo test --features backend-tract --lib
```

### Run Integration Tests
```bash
cargo test --features backend-tract --test e2e_test
```

### Run All Tests
```bash
cargo test --features backend-tract --all
```

### Expected Test Coverage
- **Storage tests**: Create/persist/remove embeddings
- **ML tests**: Cosine similarity, normalization, embedding ops
- **Config tests**: TOML parsing, defaults, GPU config
- **E2E tests**: Full auth flow simulation

## Test Results Will Show

After installing deps and running `cargo test --features backend-tract`:

```
running X tests
test storage::test_create_and_store ... ok
test ml::test_cosine_similarity ... ok
test config::test_default_config ... ok
...

test result: ok. X passed; 0 failed
```

