import React, { useEffect, useState } from "react";
import { ComposedChart, Bar } from "recharts";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  Brush,
  BarChart,
  Cell
} from "recharts";

function App() {
  const cardStyle = {
    background : "#1e293b",
    color : "#e2e8f0",
    padding : "20px",
    borderRadius : "12px",
    boxShadow : "0 4px 12px rgba(0, 0, 0, 0.3)",
    textAlign : "center",
    marginBottom : "20px",
    minHeight : "140px",
    display : "flex",
    flexDirection : "column",
    justifyContent : "center"
  }
  const [dailySummary, setDailySummary] = useState([]);
  const [intradayData, setIntradayData] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [hypothesis, setHypothesis] = useState(null);
  const [rangeData, setRangeData] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const getStats = (arr) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    return {
      min : sorted[0],
      max : sorted[sorted.length - 1],
      median : sorted[mid]
    };
  };

  const CustomTooltip = ({active, payload, label}) => { 
    if (active && payload && payload.length){ 
      return ( 
      <div style = {{ 
        background : "#1e293b", 
        padding : "10px", 
        borderRadius : "8px", 
        border : "1px solid #334155" 
      }}> 
        <p style = {{color : "#e2e8f0"}}>{label}</p> 
        <p style = {{color : "#60a5fa"}}> 
          Price : {payload[0]?.value} 
        </p> 
        <p style = {{color : "#f59e0b"}}> 
          Volatility : {payload[1]?.value} 
        </p> 
      </div> 
    ); 
  } 
  return null; 
};

  const ActivityToolTip = ({active, payload, label}) => {
    if(active && payload && payload.length){
      const vol = payload.find(p => p.dataKey === "rolling_vol_15");
      const range = payload.find(p => p.dataKey === "range");

      return (
        <div style = {{
          background : "#1e293b",
          padding : "10px",
          borderRadius : "8px",
          border : "1px solid #334155",
          color : "#e2e8f0"
        }}>
          <p>Index: {label}</p>
          <p style = {{color : "#f59e0b"}}>Volatility : {vol?.value !== undefined ? vol.value.toFixed(4) : "NA"}</p>
          <p style = {{color : "#22c55e"}}>Range : {range?.value !== undefined ? range.value.toFixed(4) : "NA"}</p>
        </div>
      );
    }
    return null;
  };
  const CandleTooltip = ({active, payload}) => {
    if(active && payload && payload.length){
      const data = payload[0].payload;
      return (
        <div style = {{
          background : "#1e293b",
          padding : "10px",
          borderRadius : "8px",
          border : "1px solid #334155",
          color : "#e2e8f0"
        }}>
          <p><stong>Index :</stong> {data.index}</p>
          <p style = {{color : "#60a5fa"}}>Open: {data.open}</p>
          <p style = {{color : "#22c55e"}}>High: {data.high}</p>
          <p style = {{color : "#ef4444"}}>Low: {data.low}</p>
          <p style = {{color : "#f59e0b"}}>Close: {data.close}</p>
        </div>
      );
    }
    return null;
  };

  const RangeToolTip = ({active, payload}) => {
    if(active && payload && payload.length){
      const data = payload[0].payload;
      return (
        <div style = {{
          background : "#1e293b",
          padding : "10px",
          borderRadius : "8px", 
          border : "1px solid #334155",
          color : "#e2e8f0"
        }}>
          <p><strong>{data.type}</strong></p>
          <p style = {{
            color : "#60a5fa"
          }}>
            Avg Range: {data.value.toFixed(2)}
          </p>
        </div>
      );
    }
    return null;
  };

  const Candlestick = (props) => {
  const { x, y, width, height, payload } = props;

  const open = payload.open;
  const close = payload.close;

  const isGreen = close >= open;

  return (
    <g>
      {/* wick of candle*/}
      <line
        x1={x + width / 2}
        x2={x + width / 2}
        y1={y}
        y2={y + height}
        stroke={isGreen ? "#22c55e" : "#ef4444"}
        strokeWidth={1}
      />

      {/* Body of candle */}
      <rect
        x={x}
        y={y + height / 4}
        width={width}
        height={height / 2}
        fill={isGreen ? "#22c55e" : "#ef4444"}
      />
    </g>
  );

};

  // DAILY DATA
  useEffect(() => {
    fetch("https://vdiim.onrender.com/daily-summary?limit=20")
      .then((res) => res.json())
      .then((data) => setDailySummary(data))
      .catch((err) => console.error(err));
  }, []);

  // INTRADAY DATA
  useEffect(() => {
    fetch("https://vdiim.onrender.com/volatility?limit=300")
      .then((res) => res.json())
      .then((data) => {
        const formatted = data.map((d, i) => ({
          index: i,
          time: new Date(d.timestamp).toLocaleTimeString(),
          open : Number(d.open),
          high : Number(d.high),
          low : Number(d.low),
          close: Number(d.close),
          rolling_vol_15: Number(d.rolling_vol_15) * 1000, // scale
          range: Number(d.range),
          vol_spike: d.vol_spike,
          ema_8: Number(d.ema_8),
          ema_30: Number(d.ema_30),
        }));

        setIntradayData(formatted);
      })
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
  if (!selectedDate) return;

  fetch(`https://vdiim.onrender.com/intraday-by-date?date=${selectedDate}`)
    .then((res) => res.json())
    .then((data) => {
      const formatted = data.map((d, i) => ({
        index: i,
        time: new Date(d.timestamp).toLocaleTimeString(),
        open : Number(d.open),
        high : Number(d.high),
        low : Number(d.low),
        close: Number(d.close),
        rolling_vol_15: Number(d.rolling_vol_15) * 1000,
        range: Number(d.range),
        vol_spike: d.vol_spike,
        ema_8: Number(d.ema_8),
        ema_30: Number(d.ema_30),
      }));

      setIntradayData(formatted);
    })
    .catch((err) => console.error(err));
}, [selectedDate]);

  useEffect(() =>{
    fetch("https://vdiim.onrender.com/hypothesis-test")
    .then(res => res.json())
    .then(data => setHypothesis(data))
    .catch(err => console.error(err));
  },[]);

  useEffect(() => {
    fetch("https://vdiim.onrender.com/range-distribution")
    .then(res => res.json())
    .then(data => setRangeData(data))
    .catch(err => console.error(err));
  }, []);

  useEffect(() => {
    fetch("https://vdiim.onrender.com/predict-next-open")
    .then(res => res.json())
    .then(data => setPrediction(data))
    .catch(err => console.error(err));
  }, []);

  return (
    <div style = {{ 
      padding : "20px",
      fontFamily : "Arial",
      backgroundColor : "#0f172a",
      color : "#e2e8f0",
      minHeight : "100vh"
      }}>
      <h1 style = {{color : "#f8fafc"}}>Nifty 50 Volatility Dashboard</h1>

      <div style = {{
        display : "grid",
        gridTemplateColumns : "repeat(auto-fit, minmax(220px, 1fr))",
        gap : "20px",
        marginBottom : "20px"
        }}>
        <div style = {cardStyle}>
          <h3>Total Days</h3>
          <p style = {{fontSize : "22px"}}>{dailySummary.length}</p>
        </div>

        <div style = {cardStyle}>
          <h3>High Vol Days</h3>
          <p style = {{color : "#ef4444", fontSize : "22px"}}>
            {dailySummary.filter(d => d.regime === "High Vol").length}
          </p>
        </div>

        <div style ={cardStyle}>
          <h3>Low Vol Days</h3>
          <p style = {{color : "#22c55e", fontSize : "22px"}}>
            {dailySummary.filter(d => d.regime === "Low Vol").length}
          </p>
        </div>

        {hypothesis &&(
          <div style = {cardStyle}>
            <h3>Hypothesis Testing</h3>        
              <p>High Vol Avg Range : {hypothesis.high_vol_mean_range.toFixed(2)}</p>
              <p>Normal Avg Range : {hypothesis.normal_vol_mean_range.toFixed(2)}</p>
              <p>p-value : {hypothesis.p_value.toFixed(4)}</p>
              <p style = {{
                marginTop : "5px",
                fontWeight : "bold",
                color : hypothesis.p_value < 0.05 ? "#22c55e" : "#ef4444"}}>
                {hypothesis.p_value < 0.05 ? "Statistically significant ✅" : "Not significant ❌"}
              </p>
          </div>
        )}
      </div>

      <h2 style = {{color : "#e2e8f0"}}>Daily Volatility Trend</h2>
      <div style = {{
        background : "white",
        padding : "15px",
        borderRadius : "10px",
        boxShadow : "0 2px 8px rgba(0, 0, 0, 0.1)",
        marginBottom : "20px"
      }}>
        {/* DAILY TREND */}
        {dailySummary.length > 0 && (
          <div style={{ width: "100%", height: 400 }}>
            <ResponsiveContainer>
              <LineChart data={dailySummary}>
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(t) =>
                    new Date(t).toLocaleDateString()
                  }
                />

                <YAxis />

                <Tooltip
                  labelFormatter={(t) =>
                    new Date(t).toLocaleDateString()
                  }
                  contentStyle={{ background: "#1e293b", border: "none" }}
                />
              
                <Line type = "natural" dataKey="avg_vol" stroke="#38bdf8" dot = {false} isAnimationActive = {true} strokeWidth = {2} animationDuration={1200}/>
                <Line type = "natural" dataKey="max_vol" stroke="#f43f5e" dot = {false} isAnimationActive = {true} strokeWidth = {2} animationDuration={1200}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <h2 style = {{color : "#e2e8f0"}}>Intraday Price + Volatility</h2>
      <div style = {{
        background : "white",
        padding : "15px",
        borderRadius : "10px",
        boxShadow : "0 2px 8px rgba(0, 0, 0, 0.1)",
        marginBottom : "20px"
      }}>
        <h2 style = {{color : "#1e293b"}}>Select Date for Intraday View</h2>

        <input
          type="date"
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          style = {{
            padding : "10px",
            borderRadius : "6px",
            border : "none",
            background : "#576375",
            color : "white",
            marginBottom : "20px"
          }}
        />

        <div style = {cardStyle}>
        {/* INTRADAY CHART */}
        {intradayData.length > 0 && (
          <div style={{ width: "100%", height: 400 }}>
            <ResponsiveContainer>
              <LineChart data={intradayData} margin = {{top : 10, right : 30, left : 0, bottom : 0}}>
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis 
                dataKey = "time" 
                domain = {['dataMin', 'dataMax']}
                tick = {{fontSize : 10}}
                />

                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" domain = {['auto','auto']} orientation="right" />

                <Tooltip 
                  content={<CustomTooltip />}
                  cursor={{ stroke: "#94a3b8", strokeWidth: 1, strokeDasharray : "3 3" }}
                  wrapperStyle={{outline : "none"}}
                  shared = {true}
                />

                <Line yAxisId="right" type="natural" dataKey="close" stroke="#60a5fa" strokeWidth={2} dot={false} activeDot = {{r : 6}} isAnimationActive = {false}/>
                <Line yAxisId="left" type="natural" dataKey="rolling_vol_15" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot = {{r : 6}} isAnimationActive = {false}/>
                <Line yAxisId="left" type="natural" dataKey="range" stroke="#22c55e" strokeWidth={2} dot={false} activeDot = {{r : 6}} isAnimationActive = {false}/>

                {/* SPIKES */}
                <Scatter
                  yAxisId = "left"
                  data={intradayData.filter((d) => d.vol_spike)}
                  dataKey = "rolling_vol_15" 
                  fill="red"
                  shape = "circle"
                  isAnimationActive = {false}
                />

                {/* ZOOM */}
                <Brush dataKey="time" height={30} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        </div>
      </div>
      
      <h2  style = {{color : "#e2e8f0"}}>Activity vs Volatility</h2>
      <div style = {{
        background : "white",
        padding : "15px",
        borderRadius : "10px",
        boxShadow : "0 2px 8px rgba(0, 0, 0, 0.1)",
        marginBottom : "20px"
      }}>
        {/* ACTIVITY VS VOL */}
        {intradayData.length > 0 && (
          <div style={{ width: "100%", height: 400 }}>
            <ResponsiveContainer>
              <LineChart data={intradayData}>
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis dataKey="index" />

                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />

                <Tooltip 
                content = {<ActivityToolTip />}
                cursor = {{stroke : "#94a3b8", strokeWidth : 1, strokeDasharray : "3 3"}}/>

                {/* VOLATILITY */}
                <Line
                  yAxisId="left"
                  type="natural"
                  dataKey="rolling_vol_15"
                  stroke="orange"
                  dot={false}
                />

                {/* ACTIVITY (RANGE) */}
                <Line
                  yAxisId="right"
                  type="natural"
                  dataKey="range"
                  stroke="green"
                  dot={false}
                />
                <Brush dataKey="index" height={30} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <h2  style = {{color : "#e2e8f0"}}>Candlestick Chart</h2>
      <div style = {{
        background : "white",
        padding : "15px",
        borderRadius : "10px",
        boxShadow : "0 2px 8px rgba(0, 0, 0, 0.1)",
        marginBottom : "20px"
      }}>
        <h2 style = {{color : "#1e293b"}}>Select Date for Intraday View</h2>

        <input
          type="date"
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          style = {{
            padding : "10px",
            borderRadius : "6px",
            border : "none",
            background : "#576375",
            color : "white",
            marginBottom : "20px"
          }}
        />

        <div style={{ width: "100%", height: 400 }}>
          <ResponsiveContainer>
            <ComposedChart data={intradayData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis domain = {['dataMin', 'dataMax']}/>
              <Tooltip content = {<CandleTooltip />} />

              <Bar
                dataKey="close"
                shape={<Candlestick />}
                barSize = {5}
              />

              <Line type="monotone" dataKey="ema_8" stroke="#facc15" dot={false}/>
              <Line type="monotone" dataKey="ema_30" stroke="#8b5cf6" dot={false} />

              <Brush dataKey="index" height={30} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <h2 style = {{color : "#e2e8f0"}}>Volatility Summary Table</h2>
      <div style = {{
        background : "#1e293b",
        color : "#e2e8f0",
        padding : "15px",
        borderRadius : "10px",
        boxShadow : "0 2px 8px rgba(0, 0, 0, 0.1)",
        marginBottom : "20px"
      }}>
        {/* Table */}
        <table cellPadding="5" style = {{
          width : "100%",
          borderCollapse : "collapse",
          background : "#1e293b"
        }}>
          <thead>
            <tr style = {{background : "#334155", color : "#f1f5f9"}}>
              <th style = {{ background : "#334155", padding : "8px"}}>Date</th>
              <th style = {{ background : "#334155", padding : "8px"}}>Average Volatility</th>
              <th style = {{ background : "#334155", padding : "8px"}}>Max Volatility</th>
              <th style = {{ background : "#334155", padding : "8px"}}>Spike Count</th>
              <th style = {{ background : "#334155", padding : "8px"}}>Regime</th>
            </tr>
          </thead>
          <tbody>
            {dailySummary.map((row, index) => (
              <tr key={index} style = {{background : index % 2 === 0 ? "#1e293b" : "#0f172a"}}
                onMouseEnter={(e) => e.currentTarget.style.background = "#334155"}
                onMouseLeave={(e) => e.currentTarget.style.background = index % 2 === 0 ? "#1e293b" : "#0f172a"}
                >
                <td style = {{ padding : "10px", textAlign : "center"}}>{new Date(row.timestamp).toLocaleDateString()}</td>
                <td style = {{ padding : "10px", textAlign : "center"}}>{row.avg_vol.toFixed(6)}</td>
                <td style = {{ padding : "10px", textAlign : "center"}}>{row.max_vol.toFixed(6)}</td>
                <td style = {{ padding : "10px", textAlign : "center"}}>{row.spike_count}</td>
                <td style={{
                    padding : "8px",
                    textAlign : "center",
                    fontWeight : "bold",
                    color:
                      row.regime === "High Vol" ? "#ef4444" :
                      row.regime === "Low Vol" ? "#22c55e" : "#e2e8f0"
                  }}>{row.regime}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Range Distribution (High vs Normal Vol)</h2>

      {rangeData &&(
        <div style={{
          background : "white",
          padding : "20px",
          borderRadius : "10px"
        }}>
          {(() => {
            const high = getStats(rangeData.high_vol);
            const normal = getStats(rangeData.normal_vol);

            return (
              <>
              <div style = {{marginBottom : "15px", color : "#1e293b"}}>
                <p>High Vol Median: {high.median.toFixed(2)}</p>
                <p>Normal Median: {normal.median.toFixed(2)}</p>
              </div>
              <ResponsiveContainer width = "100%" height = {300}>
                <BarChart
                data = {[
                  {type : "Normal", value : hypothesis.normal_vol_mean_range},
                  {type : "High Vol", value : hypothesis.high_vol_mean_range}
                ]}
                >
                  <CartesianGrid strokeDasharray = "3 3" />
                  <XAxis dataKey = "type" />
                  <YAxis />
                  <Tooltip content = {<RangeToolTip />} 
                  cursor={{ fill: "rgba(148,163,184,0.1)" }}/>

                  <Bar 
                    dataKey = "value"
                    radius={[6, 6, 0, 0]}
                    activeBar = {{fill : "#facc15"}} >
                    <Cell fill = "#22c55e" />
                    <Cell fill = "#ef4444" />
                  </Bar>
                  </BarChart>
              </ResponsiveContainer>
              </>
            );
          })()}
        </div>
      )}

      {prediction && (
        <div style = {cardStyle}>
          <h3>Next Candle Prediction</h3>
          <p>
            Predicted Open: {prediction?.predicted_next_open !== undefined ? prediction.predicted_next_open.toFixed(2) : "Loading..."}
          </p>
          <p>
            MAE: {prediction?.avg_mae !== undefined ? prediction.avg_mae.toFixed(4) : "Loading..."}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;