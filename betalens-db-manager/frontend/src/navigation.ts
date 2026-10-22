import {useCallback, useEffect, useState} from 'react';

export const pages = [
  {label: '本地表', path: '/tables'},
  {label: '更新与审计', path: '/tasks'},
  {label: '字段映射', path: '/mappings'},
  {label: '订阅源', path: '/sources'},
  {label: '配置', path: '/settings'},
  {label: '自动更新', path: '/schedules'},
];
const tablePages = new Set(['本地表', '更新与审计', '字段映射', '自动更新']);
export function pageUrl(label: string, table?: string) {
  const path = pages.find(p => p.label === label)?.path || '/tables';
  return path + (table && tablePages.has(label) ? '?table=' + encodeURIComponent(table) : '');
}
function readRoute() {
  const path = location.pathname.replace(/\/+$/, '') || '/';
  const page = pages.find(p => p.path === path)?.label || '本地表';
  return {page, table: new URLSearchParams(location.search).get('table') || undefined};
}
export function useManagerRoute() {
  const [route, setRoute] = useState(readRoute);
  useEffect(() => {
    if (location.pathname === '/') history.replaceState(null, '', pageUrl(route.page, route.table));
    const changed = () => setRoute(readRoute());
    window.addEventListener('popstate', changed);
    return () => window.removeEventListener('popstate', changed);
  }, []);
  const navigate = useCallback((page: string, table?: string) => {
    const url = pageUrl(page, table);
    if (location.pathname + location.search !== url) history.pushState(null, '', url);
    setRoute(readRoute());
  }, []);
  return {route, navigate};
}
