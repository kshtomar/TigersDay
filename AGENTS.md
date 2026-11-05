## Style

Write minimal code
Abstract when it makes the code cleaner
Constants should not be CLI flags
Single source of truth
Everything in the right home
Don't repeat yourself
Pass only necessary arguments
Never make unrelated changes
Understand game rules and properties
Backward incompatible by default

## Errors

Fail fast and loud
Leave it to standard errors
Don't check for the impossible

## Avoid

Hashing
Try except
Raise

## Comments

Never delete human written comments
Write mostly self documenting code
Explain nonobvious functionality

## Naming

Keep variable names concise
Abstract data layout behind names
Good function name obviates docstring

## Modules

Level of caution varies by module
Due to blast radius
How is feedback received?
> render, benchmark, playtest

/public - low risk, commit on summary
/ai - medium risk, read every line
/game - high risk, write by hand