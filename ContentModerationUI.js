import React, { useState } from 'react';
import { AlertCircle, AlertTriangle, CheckCircle } from 'lucide-react';

const ContentModerationUI = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setError(`An error occurred while analyzing the text: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const mitigateText = (text) => {
    const offensiveWords = [
      "shit", "fuck", "bitch", "asshole", "bastard", "crap", "dick", "cunt",
      "nazi", "retard", "slut", "whore", "idiot", "moron","fucker","motherfucker","kill"
    ];
    let mitigatedText = text;
    
    offensiveWords.forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'gi');
      mitigatedText = mitigatedText.replace(regex, '[redacted]');
    });
  
    return mitigatedText;
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 p-6">
      <div className="max-w-lg w-full bg-white rounded-lg shadow-2xl p-8">
        <h1 className="text-4xl font-extrabold text-gray-800 mb-8 text-center">Content Moderation</h1>
        <form onSubmit={handleSubmit}>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
            className="w-full mb-6 p-4 border border-gray-300 rounded-lg focus:outline-none focus:ring-4 focus:ring-pink-500"
            rows={5}
          />
          <button 
            type="submit" 
            disabled={isLoading || !text}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold py-3 rounded-lg hover:from-purple-700 hover:to-pink-700 transition disabled:opacity-50"
          >
            {isLoading ? 'Analyzing...' : 'Analyze'}
          </button>
        </form>
        {error && (
          <div className="mt-4 p-4 rounded bg-yellow-100 box">
            <AlertTriangle className="inline-block mr-2 text-yellow-700" />
            <span className="font-bold text-yellow-700">Error</span>
            <p>{error}</p>
          </div>
        )}
        {result && (
          <div className={`mt-6 p-6 rounded-lg box`}>
            <AlertCircle className={`inline-block mr-2 ${result.classification === "Not hate speech" ? "text-green-700" : "text-red-700"}`} />
            <span className="font-bold">{result.classification}</span>
            <p className="mt-2">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            <p className="mt-1">Probability of hate speech: {(result.probability_hate_speech * 100).toFixed(2)}%</p>
            <div className="mt-6 p-4 rounded bg-gray-100 box">
              <CheckCircle className="inline-block mr-2 text-gray-700" />
              <span className="font-bold">Original Text:</span>
              <p className="mt-2">{text}</p>
            </div>
            <div className="mt-4 p-4 rounded bg-gray-100 box">
              <CheckCircle className="inline-block mr-2 text-gray-700" />
              <span className="font-bold">Mitigated Text:</span>
              <p className="mt-2">{mitigateText(text)}</p>
            </div>
            {result.classification === "Hate speech" && (
              <div className="mt-4 p-4 rounded bg-yellow-100 box">
                <AlertTriangle className="inline-block mr-2 text-yellow-700" />
                <span className="font-bold">Warning:</span>
                <p className="mt-2">This content has been flagged for containing potentially harmful language.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ContentModerationUI;
