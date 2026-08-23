# Frontend delivery contract

`web/` is an ignored deployment artifact served by
`deploy/mobile-deep-links/nginx.mobile-deep-links.conf`; it is not frontend
source code and must never be edited by hand.

The Flutter/Web source repository and its build job must publish a versioned
artifact containing `index.html`, `assets/`, and a build revision.  Deployment
retrieves that artifact, verifies its checksum, and expands it into
`frontend/web/`.  API-route removals are therefore gated on a frontend release
that no longer calls those routes.

This backend repository deliberately does not fabricate a frontend source
tree.  Until the source repository is supplied, `frontend/web/` remains a
local deployment input and is excluded from Git and Docker build contexts.
