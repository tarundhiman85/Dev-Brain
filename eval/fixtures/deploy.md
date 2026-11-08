# Eval fixture: release procedure

Canary releases begin at 5 percent of traffic. If the five-minute error rate exceeds 2 percent, the release controller rolls back automatically. The on-call engineer reviews the deployment before traffic reaches 50 percent.
