## Description
- *What is being changed ?*
- *Why is this change needed ?*
- *Is it a new model project or an update to an existing one?*

## Checklist

### General
- [ ] PR title starts with the issue id in brackets (e.g. [CEPP-XXX])
- [ ] PR addresses a single issue or feature
- [ ] Commit history is clear and reasonably sized

### Open Access Procedure
*Strikethrough (~~example~~) the non-relevant items of the list and check the boxes of the completed tasks. Unchecked boxes should highlight that the procedure is still not completed and the PR cannot be merged.*

#### New model project:
- [ ] The license(s) is/are chosen

#### New and existing model projects:
- [ ] The code follows the Development Quality Standard (see [Code Quality](#code-quality))
- [ ] Patentability is checked with IP/legal team
- [ ] License and legal obligations are checked (see [Legal compliance](https://github.com/AI4SIM/governance/blob/main/open_access_procedure.md#legal-compliance))
- [ ] Known vulnerabilities are handled or documented
- [ ] The Open Access form is fulfilled and the meeting is scheduled
- [ ] The Open Access decision is taken
- [ ] Synchronized with GBL Marketing/Communication to communicate about the code open sourced

### Code Quality
*Strikethrough (~~example~~) the non-relevant items of the list and check the boxes of the completed ones. Unchecked boxes should highlight that the PR is still in draft mode.*

#### Implementation
- [ ] This code change accomplishes what it is supposed to do
- [ ] The solution cannot be simplified
- [ ] The code is at the right abstraction level
- [ ] The code is modular enough
- [ ] No better solution can be found in terms of maintainability, readability, performance, or security

#### Logic Errors and Bugs
- [ ] No use case exists in which the code does not behave as intended
- [ ] No inputs or external events could break the code

#### Error Handling and Logging
- [ ] Error handling is done the correct way
- [ ] Logging or debugging information is appropriate (no unnecessary additions/removals)

#### Dependencies
- [ ] Updates to documentation, configuration, or readme files were made as required
- [ ] No potential impacts on other parts of the system or backward compatibility

#### Security and Data Privacy
- [ ] This code change does not reveal any secret information like keys, passwords, or usernames

#### Performance
- [ ] There is no potential to significantly improve the performance of the code

#### Usability and Accessibility
- [ ] The API is well documented

#### Testing and Testability
- [ ] Automated tests have been added or updated to cover the change
- [ ] Existing tests reasonably cover the code change (unit/functional/system tests)

#### Readability
- [ ] The code is easy to understand
- [ ] Readability cannot be improved with different function, method, or variable names
- [ ] No redundant or outdated comments exist
- [ ] No commented-out code exists

#### Experts' Opinion
- [ ] A specific expert (e.g., security or usability) should review the code before acceptance: *mention them here*
