# Database connection pooling

DevBrain uses SQLAlchemy connection pooling with pool_pre_ping to detect stale
PostgreSQL connections. Each request uses a short-lived session and returns its
connection to the pool when finished.
