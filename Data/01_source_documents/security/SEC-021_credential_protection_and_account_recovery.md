# Credential Protection and Account Recovery

Document ID: SEC-021
Organization: NexaFlow Technologies
Status: Active
Version: 3.0
Effective date: 2026-01-01
Category: security
Owner: Security

## Purpose
This controlled source document defines NexaFlow Technologies requirements for credential protection and account recovery. It is designed to give a clear company reference for protection of systems, identities, information, and evidence from unauthorized access, misuse, or disruption. The document is intentionally explicit about what is required, what is merely permitted, where approval is needed, and how uncertainty should be handled so that employees and company systems do not invent policy details that are not supported by an approved source.

## Scope
This document applies to employees, contractors, administrators, and Security whenever work falls within the credential protection and account recovery domain. It should be read together with relevant security, privacy, contractual, employment, financial, product, and legal requirements. A customer contract or applicable law may impose an additional or stricter requirement for a specific situation. When that happens, the stricter or specifically applicable requirement should be followed without rewriting the baseline NexaFlow policy for unrelated cases.

## Policy and operating information
1. Passwords, recovery codes, API secrets, authentication tokens, and other credentials must be treated as sensitive security information.
2. Credentials must not be shared between users or sent through unapproved communication channels.
3. Employees must not approve unexpected MFA prompts and must report suspicious authentication activity promptly.
4. A lost or unavailable MFA device must be handled through the approved account-recovery or IT-support process rather than by bypassing MFA.
5. Identity verification is required before support personnel reset or recover a user account.
6. Privileged credentials must use approved credential-management methods and must not be stored in personal notes or unapproved files.
7. Suspected credential exposure requires prompt reporting and appropriate credential rotation or revocation by authorized personnel.
8. Support personnel must not reset another person's account solely on the basis of an unverified message claiming urgency or authority.
9. Service credentials should be scoped to the minimum required privileges and reviewed when ownership or system purpose changes.
10. Account-recovery records should identify the request, verification method, action taken, and responsible support or security owner.

## Detailed interpretation
### Rule 1
Passwords, recovery codes, API secrets, authentication tokens, and other credentials must be treated as sensitive security information.

Apply Rule 1 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

### Rule 2
Credentials must not be shared between users or sent through unapproved communication channels.

Apply Rule 2 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission. Required approval should come from the authorized role and be recorded when the process requires evidence.

### Rule 3
Employees must not approve unexpected MFA prompts and must report suspicious authentication activity promptly.

Apply Rule 3 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission. Report known facts through the approved route without delaying the report to perform unauthorized investigation.

### Rule 4
A lost or unavailable MFA device must be handled through the approved account-recovery or IT-support process rather than by bypassing MFA.

Apply Rule 4 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation. Required approval should come from the authorized role and be recorded when the process requires evidence.

### Rule 5
Identity verification is required before support personnel reset or recover a user account.

Apply Rule 5 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

### Rule 6
Privileged credentials must use approved credential-management methods and must not be stored in personal notes or unapproved files.

Apply Rule 6 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission. Required approval should come from the authorized role and be recorded when the process requires evidence.

### Rule 7
Suspected credential exposure requires prompt reporting and appropriate credential rotation or revocation by authorized personnel.

Apply Rule 7 as written and preserve its conditions, qualifiers, approvals, and limits. Required approval should come from the authorized role and be recorded when the process requires evidence. Report known facts through the approved route without delaying the report to perform unauthorized investigation.

### Rule 8
Support personnel must not reset another person's account solely on the basis of an unverified message claiming urgency or authority.

Apply Rule 8 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission.

### Rule 9
Service credentials should be scoped to the minimum required privileges and reviewed when ownership or system purpose changes.

Apply Rule 9 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

### Rule 10
Account-recovery records should identify the request, verification method, action taken, and responsible support or security owner.

Apply Rule 10 as written and preserve its conditions, qualifiers, approvals, and limits. Records should remain accurate and must not be falsified, concealed, or removed to change the apparent history of an action.

## Operating workflow
The normal operating approach for credential protection and account recovery is evidence-driven. First, identify the specific request, event, customer case, employee need, system change, or decision that is being handled. Second, identify which numbered requirement in this document actually applies. Third, confirm any relevant eligibility condition, approval, contract term, data classification, access restriction, or numerical threshold before action is taken. Fourth, perform the approved action using the responsible team or company system. Fifth, create or update the record needed for later review when the action is material. Finally, escalate questions that the available policy does not answer rather than turning an assumption into a company rule.

Teams should avoid treating the credential protection and account recovery process as a reason to bypass another control. For example, an urgent customer request does not automatically remove a security requirement, an employee request does not create a benefit that is not documented, and a manager statement does not automatically change a financial or access threshold. The correct path is to identify the governing source and the authorized exception route, if one exists.

## Worked scenarios
### Scenario 1
A practical credential protection and account recovery case is reviewed. Rule 1 states: Passwords, recovery codes, API secrets, authentication tokens, and other credentials must be treated as sensitive security information. The team should apply the documented rule to the known facts and treat any unanswered extra point as unspecified.

### Scenario 2
A practical credential protection and account recovery case is reviewed. Rule 2 states: Credentials must not be shared between users or sent through unapproved communication channels. A proposed shortcut that conflicts with this wording should be declined and routed through the approved process or a documented exception path.

### Scenario 3
A practical credential protection and account recovery case is reviewed. Rule 3 states: Employees must not approve unexpected MFA prompts and must report suspicious authentication activity promptly. A proposed shortcut that conflicts with this wording should be declined and routed through the approved process or a documented exception path.

### Scenario 4
A practical credential protection and account recovery case is reviewed. Rule 4 states: A lost or unavailable MFA device must be handled through the approved account-recovery or IT-support process rather than by bypassing MFA. The request should not proceed on assumed approval; the required authorization should be confirmed through the normal process.

### Scenario 5
A practical credential protection and account recovery case is reviewed. Rule 5 states: Identity verification is required before support personnel reset or recover a user account. The team should apply the documented rule to the known facts and treat any unanswered extra point as unspecified.

## Decision and communication guidance
When communicating about credential protection and account recovery, personnel should separate confirmed NexaFlow requirements from assumptions, estimates, and information that is simply not present in the available policy. A statement such as 'the available policy does not specify that point' is preferable to inventing a benefit, price, permission, exception, deadline, role, or guarantee. Where this document explicitly says an action is prohibited, the response should say that the action is not permitted under the normal policy rather than describing the matter as merely unknown.

False premises should be corrected when the controlled source contradicts them. If a requester asserts that a different threshold, entitlement, permission, or exception exists, the responsible team should use the documented requirement and identify the applicable approved exception or contract only when evidence for it is available. This keeps customer, employee, security, financial, and product decisions consistent across teams.

## Responsibilities
- Security owns the baseline requirements in this document and coordinates material changes.
- Managers make sure relevant personnel understand the requirements that apply to their work and do not create undocumented local exceptions.
- Employees and contractors use approved processes, protect information according to sensitivity, and raise unclear cases to the responsible function.
- System and process owners maintain enough evidence for material approvals, changes, incidents, customer commitments, or exceptions to be reviewed later.
- Security, Privacy, Finance, Legal, People Operations, Product, Engineering, and other specialist functions are involved when their controlled requirements are materially affected.

## Records and evidence
Records created for credential protection and account recovery should be accurate, attributable, and proportionate to the risk and importance of the activity. A useful record normally identifies what was requested or observed, the relevant decision, the responsible owner, required approval where applicable, and any follow-up action. Records must not be falsified, selectively altered, or removed in order to create a misleading history. Sensitive records remain subject to access-control, classification, privacy, and retention requirements.

## Exceptions and escalation
A material exception to the normal credential protection and account recovery process requires authorization from Security or another policy owner who is explicitly empowered to approve the exception. The record should state the reason, scope, duration where relevant, and any compensating action. Urgency, seniority, customer pressure, convenience, or a verbal statement from an unrelated person does not by itself establish an exception. If the policy is silent or two controlled sources appear to conflict, the matter should be escalated rather than resolved by guessing.

## Related internal topics
- Access controls may affect how this document is applied in a particular case.
- Security monitoring may affect how this document is applied in a particular case.
- Incident reporting may affect how this document is applied in a particular case.
- Security exceptions may affect how this document is applied in a particular case.
- Security, privacy, contractual, and legal requirements may impose stricter controls for sensitive or customer-specific situations.

## Dataset notice
This document is synthetic source material created for the fictional NexaFlow Technologies GenAI research project. The company, policies, products, prices, addresses, benefits, examples, and operating rules are synthetic. Public materials may inspire document structure or domain coverage, but no external policy text is copied into this source document.
