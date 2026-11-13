import {useEffect, useState} from 'react';

type Props = {
  label: string;
  page: number;
  pageSize: number;
  total: number;
  onChange: (page: number) => void;
  disabled?: boolean;
  unit?: string;
  previousLabel?: string;
  nextLabel?: string;
};

export default function Pagination({label, page, pageSize, total, onChange, disabled = false, unit = '项', previousLabel = '上一页', nextLabel = '下一页'}: Props) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const [destination, setDestination] = useState(String(page));
  useEffect(() => setDestination(String(page)), [page, pages]);
  const value = Number(destination);
  const valid = destination.trim() !== '' && Number.isSafeInteger(value) && value >= 1 && value <= pages;
  const empty = total === 0;

  return <nav className="pagination" aria-label={label + '分页'} aria-busy={disabled}>
    <span aria-live="polite">共 {total} {unit} · 第 {page} / {pages} 页</span>
    <div className="pagination-buttons">
      <button type="button" disabled={disabled || empty || page <= 1} onClick={() => onChange(1)}>首页</button>
      <button type="button" disabled={disabled || empty || page <= 1} onClick={() => onChange(page - 1)}>{previousLabel}</button>
      <button type="button" disabled={disabled || empty || page >= pages} onClick={() => onChange(page + 1)}>{nextLabel}</button>
      <button type="button" disabled={disabled || empty || page >= pages} onClick={() => onChange(pages)}>末页</button>
    </div>
    <form className="pagination-jump" onSubmit={event => {event.preventDefault(); if (!disabled && !empty && valid && value !== page) onChange(value);}}>
      <label>跳至<input aria-label={label + '跳转页码'} type="number" min="1" max={pages} step="1" value={destination} disabled={disabled || empty} onChange={event => setDestination(event.target.value)}/>页</label>
      <button type="submit" disabled={disabled || empty || !valid || value === page}>跳转</button>
    </form>
  </nav>;
}
