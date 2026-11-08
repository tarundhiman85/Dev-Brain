# Eval fixture: project cache

The project lookup cache expires entries after 15 minutes. Cache keys include the tenant ID and project ID. Project updates invalidate the cache after the write commits.
