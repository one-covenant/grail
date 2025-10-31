# Bounty Guide

This guide outlines how to earn bounties on the GRAIL subnet by helping maintain network integrity.

## Eligibility Requirements

**Evidence Required, Not Hypotheses**

To receive a bounty, you must provide concrete, verifiable evidence. Hypotheses alone are not sufficient, as most turn out to be false and waste investigation time.

## Bounty Categories

### 1. Detecting Miner Cheating

Report miners engaging in fraudulent behavior by providing concrete proof.

#### Requirements

- **Concrete Evidence**: Retrieve and analyze actual rollout data or miner outputs
- **Statistical Significance**: For pattern-based claims, analyze at least 30 randomly selected rollouts
- **Reproducibility**: Evidence must be independently verifiable

#### Examples

**Rollout Copying**
- **Claim**: Miner X is copying rollouts from Miner Y
- **Evidence Required**: 
  - Retrieve rollouts from both miners
  - Show they are identical (matching token IDs, logprobs, and hidden states)
  - Include timestamps proving simultaneous or suspiciously timed submissions

**Completion Manipulation**
- **Claim**: Miner is prepending the gold answer to completions before generation to reduce completion length
- **Evidence Required**:
  - Retrieve at least 30 randomly selected rollouts from the miner
  - Demonstrate the pattern persists across the sample
  - Show structural anomalies (e.g., answer tokens appearing before reasoning)

### 2. Penetration Testing

Demonstrate exploits by replicating the cheating mechanism yourself.

#### Requirements

- **Working Exploit**: Provide complete, runnable code
- **Mainnet Proof**: Show the exploit successfully bypassing validation on mainnet
- **Duration**: Exploit must work for at least 2 consecutive hours
- **Documentation**: Clear explanation of the vulnerability and how it works

#### Example

**Verification Prediction Exploit**
- **Claim**: Miners can predict which rollouts will be verified and precompute only those
- **Proof Required**:
  1. Write code that exploits this prediction mechanism
  2. Run the exploit on mainnet
  3. Show validation logs proving it passed all checks for 2+ hours
  4. Provide the complete codebase used for the exploit
  5. Document the vulnerability and suggested fix

## Submission Process

1. **Gather Evidence**: Collect all required data and proof
2. **Document Findings**: Create a clear, structured, and short (max 4 paragraphs) report with:
   - Summary of the issue
   - Complete evidence (data files, logs, screenshots)
   - Reproduction steps/code
   - For exploits: complete code and execution logs
3. **Submit**: Contact the subnet operators with your documented findings

## What Will NOT Earn a Bounty

- Unsubstantiated hypotheses or suspicions
- Partial evidence requiring further investigation
- Reports without retrievable proof
- Exploits that fail on mainnet or work for less than 2 hours
- Already-known vulnerabilities

## Best Practices

- **Random Sampling**: When analyzing patterns, use random sampling to avoid selection bias
- **Timestamp Everything**: Include timestamps in all evidence
- **Preserve Data**: Keep copies of all rollouts and logs used in your analysis
- **Code Quality**: For penetration tests, provide clean, documented code
- **Respect Network**: Test exploits responsibly; don't harm legitimate miners

---

*Help us maintain a fair and secure subnet. Evidence-based reports are rewarded; speculation is not.*
