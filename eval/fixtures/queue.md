# Eval fixture: billing queue

The billing worker deduplicates events using the provider event ID. It acknowledges a message only after the database transaction commits. Messages that fail five attempts go to the dead-letter queue.
