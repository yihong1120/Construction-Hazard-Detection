# Frontend delivery contract

The public Flutter source is maintained in
[visionnaire-flutter](https://github.com/yihong1120/visionnaire-flutter).

`web/` is an ignored deployment artifact served by
`deploy/mobile-deep-links/nginx.mobile-deep-links.conf`; it is not frontend
source code and must never be edited by hand.

The Flutter/Web source repository and its build job must publish a versioned
artifact containing `index.html`, `assets/`, and a build revision.  Deployment
retrieves that artifact, verifies its checksum, and expands it into
`frontend/web/`.  API-route removals are therefore gated on a frontend release
that no longer calls those routes.

This backend repository deliberately does not fabricate a frontend source
tree. `frontend/web/` remains a local deployment input and is excluded from
Git and Docker build contexts.

## Backend compatibility

Build and deploy a Flutter revision that uses the current backend contract:

- Flutter Web sends playback control requests through the same-origin BFF at
  `/bff/db_management/api/playback/*`.
- Native clients send playback control requests to
  `/hazard/api/db_management/api/playback/*` after verifying their signed
  deployment profile.
- A Registry client validates HTTPS, redirect handling, JSON size and the
  Ed25519 signature. It must not reject an otherwise valid response solely
  because an intermediary omits `Cache-Control: no-store`.

Deploy the frontend artifact and this backend revision together whenever one
of these public paths or the Registry document contract changes.
