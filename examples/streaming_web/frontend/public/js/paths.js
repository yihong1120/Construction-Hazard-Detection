export function appBasePath() {
    return window.location.pathname.startsWith('/hazard') ? '/hazard' : '';
}

export function apiPath(path) {
    return `${appBasePath()}/api${path}`;
}

export function pagePath(path) {
    return `${appBasePath()}${path}`;
}

export function websocketPath(path) {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    return `${protocol}${window.location.host}${apiPath(path)}`;
}
