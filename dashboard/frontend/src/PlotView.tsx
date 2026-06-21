import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import type { Config, Data, Layout } from 'plotly.js';

const Plot = createPlotlyComponent(Plotly);

export default function PlotView({
  data,
  layout,
  config
}: {
  data: Data[];
  layout: Partial<Layout>;
  config: Partial<Config>;
}) {
  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      useResizeHandler
      className="plot"
    />
  );
}
