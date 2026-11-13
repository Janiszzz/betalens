import { createContext, useContext } from 'react';
import { api } from './api';

/** Result views only read through this interface; execution stays upstream. */
export type BacktestResultClient = Pick<typeof api, 'profiling' | 'table' | 'downloadUrl'>;
export const ResultClientContext = createContext<BacktestResultClient>(api);
export const useResultClient = () => useContext(ResultClientContext);
