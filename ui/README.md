# `ui/`

**Owner:** Frontend

The reviewer's screens, served by the API from the same address.

| Page | Purpose |
|:--|:--|
| `index.html`, `signup.html` | Sign in and create an account |
| `dashboard.html` | Overview of the current review and the PDF download |
| `upload.html` | Upload a statement and see review history |
| `findings.html` | Every finding, with Acknowledge, Flag for follow-up and Dismiss |
| `trends.html` | Ratio charts and material deviations |
| `risk.html` | The risk score and what drove it |
| `chatbot.html` | The AI summary of the review |

Scripts are in `js/` and styles in `css/`. Pages call the API with relative paths, so they work wherever the service is deployed.
