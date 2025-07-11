import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Dashboard({ setIsAuthenticated }) {
  const [prediction, setPrediction] = useState(null);
  const [timestamp, setTimestamp] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  const [prevPrediction, setPrevPrediction] = useState(null);
  const [predictionColor, setPredictionColor] = useState("text-indigo-600"); // по умолчанию


  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:8000/prediction/history', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setPredictionHistory(response.data.history);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  };

  const fetchPrediction = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:8000/prediction', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (response.data.status === 'ready') {
        const newPrediction = response.data.prediction;
        
        if (prevPrediction !== null) {
        if (newPrediction > prevPrediction) {
          setPredictionColor("text-green-600");
        } else if (newPrediction < prevPrediction) {
          setPredictionColor("text-red-600");
        } else {
          setPredictionColor("text-yellow-500");
        }
      }
      setPrevPrediction(newPrediction);
        // Обновляем текущее предсказание и временную метку
        setPrediction(newPrediction);
        setTimestamp(response.data.timestamp);
        
        // Проверяем, есть ли это значение в истории
        const exists = predictionHistory.some(p => 
          p.prediction === newPrediction
        );
        
        // Если это новое уникальное значение, обновляем историю
        if (!exists) {
          await fetchHistory();
        }
      }
      setError(null);
    } catch (err) {
      if (err.response && err.response.status === 401) {
        setIsAuthenticated(false);
        navigate('/login');
      } else {
        setError('Failed to fetch prediction');
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // При первом рендере загружаем и предсказание, и историю
    const initialLoad = async () => {
      await fetchHistory();
      await fetchPrediction();
    };
    initialLoad();

    // Устанавливаем интервал только для обновления предсказания
    const interval = setInterval(fetchPrediction, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex-shrink-0 flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">ML Prediction Service</h1>
            </div>
            <div className="flex items-center">
              <button
                onClick={handleLogout}
                className="ml-4 px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 transition-colors"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {error && (
          <div className="mb-6">
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
              <span className="block sm:inline">{error}</span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Current Prediction Card */}
          <div className="bg-white overflow-hidden shadow-lg rounded-lg">
            <div className="px-6 py-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">Current Prediction</h2>
              <div className="flex flex-col items-center justify-center">
                {prediction !== null ? (
                  <>
                    <div className={`text-5xl font-bold mb-2 ${predictionColor}`}>
                      {prediction.toFixed(4)}
                    </div>
                    <div className="text-sm text-gray-500">
                      Last updated: {timestamp ? new Date(timestamp).toLocaleString() : 'N/A'}
                    </div>
                  </>
                ) : (
                  <div className="text-xl text-gray-500">
                    {isLoading ? (
                      <div className="flex items-center">
                        <svg className="animate-spin h-8 w-8 text-indigo-600 mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Loading prediction...
                      </div>
                    ) : "No prediction available"}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Prediction History Card */}
          <div className="bg-white overflow-hidden shadow-lg rounded-lg">
            <div className="px-6 py-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">Prediction History</h2>
              <div className="overflow-y-auto max-h-[400px]">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Time
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Value
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {predictionHistory.map((pred, index) => (
                      <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {new Date(pred.timestamp).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {pred.prediction.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default Dashboard; 