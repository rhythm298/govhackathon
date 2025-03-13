import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertTriangle, ArrowDown, ArrowUp, Calendar, Clock, MapPin, Navigation, Settings, Truck, Users } from 'lucide-react';

// Mock data that would come from the Python backend
const generateMockData = () => {
  // Traffic congestion by hour
  const hourlyData = Array.from({ length: 24 }, (_, i) => ({
    hour: i,
    congestion: Math.floor(Math.random() * 100),
    prediction: Math.floor(Math.random() * 100),
  }));

  // Traffic by road segments
  const roadSegments = [
    { name: 'Main St', value: Math.floor(Math.random() * 1000) + 500 },
    { name: 'Highway 101', value: Math.floor(Math.random() * 1500) + 800 },
    { name: 'Downtown', value: Math.floor(Math.random() * 800) + 400 },
    { name: 'Residential', value: Math.floor(Math.random() * 600) + 200 },
    { name: 'Industrial', value: Math.floor(Math.random() * 700) + 300 },
  ];

  // Traffic factors
  const trafficFactors = [
    { name: 'Weather', value: Math.floor(Math.random() * 30) + 10 },
    { name: 'Time of Day', value: Math.floor(Math.random() * 30) + 10 },
    { name: 'Day of Week', value: Math.floor(Math.random() * 20) + 10 },
    { name: 'Events', value: Math.floor(Math.random() * 15) + 5 },
    { name: 'Construction', value: Math.floor(Math.random() * 10) + 5 },
  ];

  return { hourlyData, roadSegments, trafficFactors };
};

const Dashboard = () => {
  const [data, setData] = useState(generateMockData());
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [selectedTimeRange, setSelectedTimeRange] = useState('day');
  const [criticalAlerts, setCriticalAlerts] = useState([
    { id: 1, location: 'Highway 101 North', severity: 'High', message: 'Accident reported, heavy congestion expected' },
    { id: 2, location: 'Downtown Main St', severity: 'Medium', message: 'Construction causing delays' },
  ]);

  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];
  
  useEffect(() => {
    // This would be replaced with actual API calls to the Python backend
    const interval = setInterval(() => {
      setData(generateMockData());
    }, refreshInterval * 1000);
    
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const handleRefreshChange = (e) => {
    setRefreshInterval(Number(e.target.value));
  };

  const handleTimeRangeChange = (range) => {
    setSelectedTimeRange(range);
  };

  const StatCard = ({ title, value, change, icon }) => (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-sm font-medium text-gray-500">{title}</h3>
        {icon}
      </div>
      <div className="flex items-baseline">
        <p className="text-2xl font-semibold">{value}</p>
        <span className={`ml-2 text-sm ${change >= 0 ? 'text-green-500' : 'text-red-500'} flex items-center`}>
          {change >= 0 ? <ArrowUp className="h-4 w-4 mr-1" /> : <ArrowDown className="h-4 w-4 mr-1" />}
          {Math.abs(change)}%
        </span>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="bg-blue-600 text-white p-4">
        <div className="container mx-auto">
          <h1 className="text-2xl font-bold">AI-Powered Traffic Congestion Dashboard</h1>
          <p className="text-sm">Real-time predictions and analytics for intelligent traffic management</p>
        </div>
      </div>
      
      <div className="container mx-auto p-4">
        {/* Controls */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex space-x-2">
            <button 
              onClick={() => handleTimeRangeChange('day')} 
              className={`px-3 py-1 rounded ${selectedTimeRange === 'day' ? 'bg-blue-600 text-white' : 'bg-white'}`}
            >
              Day
            </button>
            <button 
              onClick={() => handleTimeRangeChange('week')} 
              className={`px-3 py-1 rounded ${selectedTimeRange === 'week' ? 'bg-blue-600 text-white' : 'bg-white'}`}
            >
              Week
            </button>
            <button 
              onClick={() => handleTimeRangeChange('month')} 
              className={`px-3 py-1 rounded ${selectedTimeRange === 'month' ? 'bg-blue-600 text-white' : 'bg-white'}`}
            >
              Month
            </button>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Refresh:</span>
            <select 
              value={refreshInterval} 
              onChange={handleRefreshChange}
              className="bg-white border border-gray-300 rounded px-2 py-1 text-sm"
            >
              <option value={10}>10s</option>
              <option value={30}>30s</option>
              <option value={60}>1m</option>
              <option value={300}>5m</option>
            </select>
          </div>
        </div>
        
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatCard 
            title="Current Congestion Index" 
            value="78/100" 
            change={5} 
            icon={<AlertTriangle className="h-5 w-5 text-yellow-500" />}
          />
          <StatCard 
            title="Active Vehicles" 
            value="12,453" 
            change={-2} 
            icon={<Truck className="h-5 w-5 text-blue-500" />}
          />
          <StatCard 
            title="Average Speed" 
            value="27 mph" 
            change={-8} 
            icon={<Navigation className="h-5 w-5 text-green-500" />}
          />
          <StatCard 
            title="Prediction Accuracy" 
            value="92%" 
            change={3} 
            icon={<Settings className="h-5 w-5 text-purple-500" />}
          />
        </div>
        
        {/* Critical Alerts */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          <h2 className="text-lg font-semibold mb-2 flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
            Critical Alerts
          </h2>
          <div className="space-y-2">
            {criticalAlerts.map((alert) => (
              <div key={alert.id} className="border-l-4 border-red-500 pl-3 py-2">
                <div className="flex justify-between">
                  <span className="font-medium text-gray-900">{alert.location}</span>
                  <span className={`text-sm px-2 py-0.5 rounded ${
                    alert.severity === 'High' ? 'bg-red-100 text-red-800' : 
                    alert.severity === 'Medium' ? 'bg-yellow-100 text-yellow-800' : 
                    'bg-blue-100 text-blue-800'
                  }`}>
                    {alert.severity}
                  </span>
                </div>
                <p className="text-sm text-gray-600">{alert.message}</p>
              </div>
            ))}
          </div>
        </div>
        
        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Traffic Prediction Chart */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Clock className="h-5 w-5 text-blue-500 mr-2" />
              Hourly Traffic Prediction
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.hourlyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Congestion Index', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="congestion" stroke="#8884d8" name="Actual Congestion" />
                <Line type="monotone" dataKey="prediction" stroke="#82ca9d" name="Predicted Congestion" strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {/* Road Segments Chart */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <MapPin className="h-5 w-5 text-green-500 mr-2" />
              Traffic by Road Segment
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.roadSegments}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Vehicle Count', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#8884d8" name="Vehicle Count">
                  {data.roadSegments.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          {/* Traffic Factors Chart */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Users className="h-5 w-5 text-purple-500 mr-2" />
              Traffic Influencing Factors
            </h2>
            <div className="flex justify-center">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={data.trafficFactors}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    nameKey="name"
                    label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {data.trafficFactors.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Prediction Performance */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Calendar className="h-5 w-5 text-orange-500 mr-2" />
              Weekly Prediction Performance
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={[
                {day: 'Mon', accuracy: 91, latency: 2.3},
                {day: 'Tue', accuracy: 93, latency: 2.1},
                {day: 'Wed', accuracy: 89, latency: 2.4},
                {day: 'Thu', accuracy: 94, latency: 1.9},
                {day: 'Fri', accuracy: 95, latency: 1.8},
                {day: 'Sat', accuracy: 87, latency: 2.5},
                {day: 'Sun', accuracy: 90, latency: 2.2},
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis yAxisId="left" label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: 'Latency (s)', angle: 90, position: 'insideRight' }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#8884d8" name="Prediction Accuracy (%)" />
                <Line yAxisId="right" type="monotone" dataKey="latency" stroke="#82ca9d" name="Response Latency (s)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Map Placeholder - In a real implementation, this would be an actual map */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center">
            <MapPin className="h-5 w-5 text-red-500 mr-2" />
            Traffic Heatmap
          </h2>
          <div className="bg-gray-200 h-64 rounded flex items-center justify-center">
            <div className="text-center">
              <MapPin className="h-12 w-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-600">Interactive traffic map would be rendered here</p>
              <p className="text-sm text-gray-500">Displaying real-time congestion data with predictive overlays</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
