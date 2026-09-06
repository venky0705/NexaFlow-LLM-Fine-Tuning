# Authentication and MFA

Document ID: SEC-001
Organization: NexaFlow Technologies
Status: Active
Version: 3.0
Effective date: 2026-01-01
Category: security
Owner: Security

## Purpose
This controlled source document defines NexaFlow Technologies requirements for authentication and mfa. It is designed to give a clear company reference for protection of systems, identities, information, and evidence from unauthorized access, misuse, or disruption. The document is intentionally explicit about what is required, what is merely permitted, where approval is needed, and how uncertainty should be handled so that employees and company systems do not invent policy details that are not supported by an approved source.

## Scope
This document applies to employees, contractors, administrators, and Security whenever work falls within the authentication and mfa domain. It should be read together with relevant security, privacy, contractual, employment, financial, product, and legal requirements. A customer contract or applicable law may impose an additional or stricter requirement for a specific situation. When that happens, the stricter or specifically applicable requirement should be followed without rewriting the baseline NexaFlow policy for unrelated cases.

## Policy and operating information
1. Multi-factor authentication is mandatory for employees accessing NexaFlow production systems.
2. Multi-factor authentication is mandatory for administrator access to corporate cloud services.
3. Passwords must not be shared between users.
4. Employees must not approve unexpected multi-factor authentication prompts.
5. Suspected credential compromise must be reported immediately to Security or Corporate IT.
6. Privileged credentials must be stored using approved credential-management methods.
7. Default vendor passwords must be changed before a system is placed into production.
8. Authentication controls must not be intentionally bypassed without an approved security exception.
9. MFA is mandatory for administrator access to corporate cloud services.
10. Employees must not approve unexpected MFA prompts.
11. Suspected credential compromise must be reported immediately.
12. Privileged credentials must use approved credential-management methods.
13. Default vendor passwords must be changed before production use.

## Detailed interpretation
### Rule 1
Multi-factor authentication is mandatory for employees accessing NexaFlow production systems.

Apply Rule 1 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

### Rule 2
Multi-factor authentication is mandatory for administrator access to corporate cloud services.

Apply Rule 2 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

### Rule 3
Passwords must not be shared between users.

Apply Rule 3 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission.

### Rule 4
Employees must not approve unexpected multi-factor authentication prompts.

Apply Rule 4 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission.

### Rule 5
Suspected credential compromise must be reported immediately to Security or Corporate IT.

Apply Rule 5 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation. Report known facts through the approved route without delaying the report to perform unauthorized investigation.

### Rule 6
Privileged credentials must be stored using approved credential-management methods.

Apply Rule 6 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation. Required approval should come from the authorized role and be recorded when the process requires evidence.

### Rule 7
Default vendor passwords must be changed before a system is placed into production.

Apply Rule 7 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

### Rule 8
Authentication controls must not be intentionally bypassed without an approved security exception.

Apply Rule 8 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission. Required approval should come from the authorized role and be recorded when the process requires evidence.

### Rule 9
MFA is mandatory for administrator access to corporate cloud services.

Apply Rule 9 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

### Rule 10
Employees must not approve unexpected MFA prompts.

Apply Rule 10 as written and preserve its conditions, qualifiers, approvals, and limits. Because this is a prohibition, urgency, seniority, customer pressure, or convenience does not turn the action into normal permission.

### Rule 11
Suspected credential compromise must be reported immediately.

Apply Rule 11 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation. Report known facts through the approved route without delaying the report to perform unauthorized investigation.

### Rule 12
Privileged credentials must use approved credential-management methods.

Apply Rule 12 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation. Required approval should come from the authorized role and be recorded when the process requires evidence.

### Rule 13
Default vendor passwords must be changed before production use.

Apply Rule 13 as written and preserve its conditions, qualifiers, approvals, and limits. Mandatory wording describes a required control rather than an optional recommendation.

## Operating workflow
The normal operating approach for authentication and mfa is evidence-driven. First, identify the specific request, event, customer case, employee need, system change, or decision that is being handled. Second, identify which numbered requirement in this document actually applies. Third, confirm any relevant eligibility condition, approval, contract term, data classification, access restriction, or numerical threshold before action is taken. Fourth, perform the approved action using the responsible team or company system. Fifth, create or update the record needed for later review when the action is material. Finally, escalate questions that the available policy does not answer rather than turning an assumption into a company rule.

Teams should avoid treating the authentication and mfa process as a reason to bypass another control. For example, an urgent customer request does not automatically remove a security requirement, an employee request does not create a benefit that is not documented, and a manager statement does not automatically change a financial or access threshold. The correct path is to identify the governing source and the authorized exception route, if one exists.

## Worked scenarios
### Scenario 1
A practical authentication and mfa case is reviewed. Rule 1 states: Multi-factor authentication is mandatory for employees accessing NexaFlow production systems. The team should apply the documented rule to the known facts and treat any unanswered extra point as unspecified.

### Scenario 2
A practical authentication and mfa case is reviewed. Rule 2 states: Multi-factor authentication is mandatory for administrator access to corporate cloud services. The team should apply the documented rule to the known facts and treat any unanswered extra point as unspecified.

### Scenario 3
A practical authentication and mfa case is reviewed. Rule 3 states: Passwords must not be shared between users. A proposed shortcut that conflicts with this wording should be declined and routed through the approved process or a documented exception path.

### Scenario 4
A practical authentication and mfa case is reviewed. Rule 4 states: Employees must not approve unexpected multi-factor authentication prompts. A proposed shortcut that conflicts with this wording should be declined and routed through the approved process or a documented exception path.

### Scenario 5
A practical authentication and mfa case is reviewed. Rule 5 states: Suspected credential compromise must be reported immediately to Security or Corporate IT. The issue should be reported through the approved route using the facts currently known.

## Decision and communication guidance
When communicating about authentication and mfa, personnel should separate confirmed NexaFlow requirements from assumptions, estimates, and information that is simply not present in the available policy. A statement such as 'the available policy does not specify that point' is preferable to inventing a benefit, price, permission, exception, deadline, role, or guarantee. Where this document explicitly says an action is prohibited, the response should say that the action is not permitted under the normal policy rather than describing the matter as merely unknown.

False premises should be corrected when the controlled source contradicts them. If a requester asserts that a different threshold, entitlement, permission, or exception exists, the responsible team should use the documented requirement and identify the applicable approved exception or contract only when evidence for it is available. This keeps customer, employee, security, financial, and product decisions consistent across teams.

## Responsibilities
- Security owns the baseline requirements in this document and coordinates material changes.
- Managers make sure relevant personnel understand the requirements that apply to their work and do not create undocumented local exceptions.
- Employees and contractors use approved processes, protect information according to sensitivity, and raise unclear cases to the responsible function.
- System and process owners maintain enough evidence for material approvals, changes, incidents, customer commitments, or exceptions to be reviewed later.
- Security, Privacy, Finance, Legal, People Operations, Product, Engineering, and other specialist functions are involved when their controlled requirements are materially affected.

## Records and evidence
Records created for authentication and mfa should be accurate, attributable, and proportionate to the risk and importance of the activity. A useful record normally identifies what was requested or observed, the relevant decision, the responsible owner, required approval where applicable, and any follow-up action. Records must not be falsified, selectively altered, or removed in order to create a misleading history. Sensitive records remain subject to access-control, classification, privacy, and retention requirements.

## Exceptions and escalation
A material exception to the normal authentication and mfa process requires authorization from Security or another policy owner who is explicitly empowered to approve the exception. The record should state the reason, scope, duration where relevant, and any compensating action. Urgency, seniority, customer pressure, convenience, or a verbal statement from an unrelated person does not by itself establish an exception. If the policy is silent or two controlled sources appear to conflict, the matter should be escalated rather than resolved by guessing.

## Related internal topics
- Access controls may affect how this document is applied in a particular case.
- Security monitoring may affect how this document is applied in a particular case.
- Incident reporting may affect how this document is applied in a particular case.
- Security exceptions may affect how this document is applied in a particular case.
- Security, privacy, contractual, and legal requirements may impose stricter controls for sensitive or customer-specific situations.

## Dataset notice
This document is synthetic source material created for the fictional NexaFlow Technologies GenAI research project. The company, policies, products, prices, addresses, benefits, examples, and operating rules are synthetic. Public materials may inspire document structure or domain coverage, but no external policy text is copied into this source document.
