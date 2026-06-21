declare module 'plotly.js-dist-min' {
  const Plotly: any;
  export default Plotly;
}

declare module 'react-plotly.js/factory' {
  import type Plot from 'react-plotly.js';

  export default function createPlotlyComponent(plotly: any): typeof Plot;
}
