# `ui/`

**Owner:** Frontend

The reviewer's screens. The app serves this folder from `/` on the same address as the API.

## Upload

- `index.html` - the first page
- your other `.html` pages, and your CSS, JavaScript and image files or folders

Call the API with relative paths such as `fetch("/review/upload")`, never `http://localhost:8000`. Uploads accept `.csv` and `.xlsx` only. Exact request and response fields are at `/docs` when the app runs.

## Never upload

`node_modules/`, build output folders, `.env`, `README.md`
