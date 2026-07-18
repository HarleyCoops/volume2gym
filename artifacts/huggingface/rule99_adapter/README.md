# Volume2Gym Railroad Proof Model Card

## Model Status

This package includes a Volume2Gym symbolic adapter generated from the
proof-cycle training tasks and extracted source rule schema. The adapter is
transparent rather than neural: `adapter_config.json` records the trainer and
source contract, while `adapter_weights.json` records rule, action, procedure,
condition, and terminology weights.

## Behavior Target

The adapter answers source-grounded technical scenarios as structured JSON plus
a final natural-language answer. It cites applicable rules, includes mandatory
safety actions, avoids forbidden unsafe actions, preserves procedure order, and
uses domain terminology.

## Current Proof Result

- Baseline mean score: 0.00
- Adapter mean score: 1.00
- Absolute delta: 1.00

## Limitations

This artifact is not a certified operating assistant and is not
railroad-specific as a method. It is an auditable proof adapter for the
Volume2Gym loop across manuals, textbooks, repair guides, codes, and technical
volumes.
