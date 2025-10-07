import React, { useState } from 'react';
import { Database, MessageSquare, Search, Brain, Zap, FileText, Filter, RotateCw, CheckCircle } from 'lucide-react';

const SystemDiagrams = () => {
  const [activeTab, setActiveTab] = useState('architecture');

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
      <h1 className="text-3xl font-bold text-center mb-6 text-indigo-900">
        Enhanced Microcontroller RAG + LINE Bot
      </h1>

      <div className="flex gap-2 mb-6 flex-wrap">
        <button onClick={() => setActiveTab('architecture')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'architecture'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'bg-white text-indigo-600 hover:bg-indigo-50'
            }`}
        >
          System Architecture
        </button>
        <button onClick={() => setActiveTab('flow')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'flow'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'bg-white text-indigo-600 hover:bg-indigo-50'
            }`}
        >
          Data Flow
        </button>
        <button onClick={() => setActiveTab('retrieval')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'retrieval'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'bg-white text-indigo-600 hover:bg-indigo-50'
            }`}
        >
          Retrieval Pipeline
        </button>
      </div>

      {activeTab === 'architecture' &&
        <ArchitectureDiagram />}
      {activeTab === 'flow' &&
        <FlowDiagram />}
      {activeTab === 'retrieval' &&
        <RetrievalDiagram />}
    </div>
  );
};

const ArchitectureDiagram = () => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-indigo-800 flex items-center gap-2">
        <Database className="w-6 h-6" />
        System Architecture
      </h2>

      <div className="space-y-6">
        {/* User Layer */}
        <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
          <div className="flex items-center gap-3 mb-3">
            <MessageSquare className="w-6 h-6 text-green-600" />
            <h3 className="font-bold text-lg text-green-800">User Interface Layer</h3>
          </div>
          <div className="flex gap-4 flex-wrap">
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-green-200">
              LINE App
            </div>
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-green-200">
              LINE Messaging API
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Application Layer */}
        <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
          <div className="flex items-center gap-3 mb-3">
            <Zap className="w-6 h-6 text-blue-600" />
            <h3 className="font-bold text-lg text-blue-800">Application Layer</h3>
          </div>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white px-4 py-2 rounded shadow-sm border border-blue-200">
                Flask Server
              </div>
              <div className="bg-white px-4 py-2 rounded shadow-sm border border-blue-200">
                Webhook Handler
              </div>
            </div>
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-blue-200 text-center">
              Query Preprocessor
            </div>
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-blue-200 text-center">
              Simple Response Handler
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* RAG Layer */}
        <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
          <div className="flex items-center gap-3 mb-3">
            <Brain className="w-6 h-6 text-purple-600" />
            <h3 className="font-bold text-lg text-purple-800">RAG Processing Layer</h3>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-white px-3 py-2 rounded shadow-sm border border-purple-200 text-sm text-center">
              Vector Retriever<br />(ChromaDB)
            </div>
            <div className="bg-white px-3 py-2 rounded shadow-sm border border-purple-200 text-sm text-center">
              BM25 Retriever
            </div>
            <div className="bg-white px-3 py-2 rounded shadow-sm border border-purple-200 text-sm text-center">
              Ensemble Retriever
            </div>
          </div>
          <div className="mt-3 bg-white px-4 py-2 rounded shadow-sm border border-purple-200 text-center">
            Cohere Reranker
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* LLM Layer */}
        <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-4">
          <div className="flex items-center gap-3 mb-3">
            <Brain className="w-6 h-6 text-orange-600" />
            <h3 className="font-bold text-lg text-orange-800">Generation Layer</h3>
          </div>
          <div className="space-y-3">
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-orange-200 text-center">
              Gemini 2.0 Flash LLM
            </div>
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-orange-200 text-center">
              Answer Formatter
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Data Layer */}
        <div className="bg-gray-100 border-2 border-gray-300 rounded-lg p-4">
          <div className="flex items-center gap-3 mb-3">
            <FileText className="w-6 h-6 text-gray-600" />
            <h3 className="font-bold text-lg text-gray-800">Data Layer</h3>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-gray-300 text-center">
              Markdown Document
            </div>
            <div className="bg-white px-4 py-2 rounded shadow-sm border border-gray-300 text-center">
              Vector Store
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const FlowDiagram = () => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-indigo-800 flex items-center gap-2">
        <RotateCw className="w-6 h-6" />
        Data Flow Process
      </h2>

      <div className="space-y-4">
        {[
          { num: 1, title: 'User sends message via LINE', color: 'green', icon: MessageSquare },
          { num: 2, title: 'Flask receives webhook', color: 'blue', icon: Zap },
          {
            num: 3, title: 'Check if simple/casual question', color: 'yellow', icon: Filter,
            sub: ['Yes → Send simple response', 'No → Continue to RAG']
          },
          {
            num: 4, title: 'Preprocess query', color: 'purple', icon: Search,
            sub: ['Remove stop words', 'Add keyword expansions', 'Map to domain terms']
          },
          {
            num: 5, title: 'Retrieve relevant documents', color: 'indigo', icon: Database,
            sub: ['Vector search (MMR)', 'BM25 keyword search', 'Ensemble combination']
          },
          {
            num: 6, title: 'Rerank results', color: 'pink', icon: RotateCw,
            sub: ['Cohere rerank top 10', 'Sort by relevance score']
          },
          {
            num: 7, title: 'Generate answer with LLM', color: 'orange', icon: Brain,
            sub: ['Format context', 'Apply prompt template', 'Generate response']
          },
          {
            num: 8, title: 'Clean and format output', color: 'red', icon: CheckCircle,
            sub: ['Remove markdown artifacts', 'Clean formatting', 'Truncate if needed']
          },
          { num: 9, title: 'Send reply to LINE user', color: 'green', icon: MessageSquare }
        ].map((step) => {
          const Icon = step.icon;
          return (
            <div key={step.num}>
              <div className={`bg-${step.color}-50 border-l-4 border-${step.color}-500 p-4 rounded-r-lg shadow-sm`}>
                <div className="flex items-center gap-3">
                  <div className={`bg-${step.color}-500 text-white w-8 h-8 rounded-full flex items-center justify-center
            font-bold`}>
                    {step.num}
                  </div>
                  <Icon className={`w-5 h-5 text-${step.color}-600`} />
                  <h3 className="font-semibold text-gray-800">{step.title}</h3>
                </div>
                {step.sub && (
                  <div className="mt-2 ml-11 space-y-1">
                    {step.sub.map((item, idx) => (
                      <div key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                        <span className="text-gray-400">•</span>
                        {item}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {step.num < 9 && (<div className="flex justify-center py-2">
                <div className="text-2xl text-gray-400">↓</div>
              </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const RetrievalDiagram = () => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-indigo-800 flex items-center gap-2">
        <Search className="w-6 h-6" />
        Enhanced Retrieval Pipeline
      </h2>

      <div className="space-y-6">
        {/* Input Query */}
        <div className="text-center">
          <div className="inline-block bg-green-100 border-2 border-green-400 rounded-lg px-6 py-3">
            <div className="font-bold text-green-800">Input Query</div>
            <div className="text-sm text-green-600 mt-1">"โครงงานวิชานี้ต้องทำอย่างไร?"</div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Preprocessing */}
        <div className="bg-yellow-50 border-2 border-yellow-300 rounded-lg p-4">
          <h3 className="font-bold text-yellow-800 mb-3">Query Preprocessing</h3>
          <div className="space-y-2 text-sm">
            <div className="bg-white p-2 rounded border border-yellow-200">
              ✓ Remove stop words: อย่างไร, ต้อง
            </div>
            <div className="bg-white p-2 rounded border border-yellow-200">
              ✓ Expand keywords: โครงงานกลุ่ม, ข้อกำหนด, เกณฑ์
            </div>
            <div className="bg-white p-2 rounded border border-yellow-200">
              <strong>Output:</strong> "โครงงาน วิชา โครงงานกลุ่มรายวิชา 5-6 คน..."
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Parallel Retrieval */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
            <h3 className="font-bold text-blue-800 mb-2 text-center">Vector Search</h3>
            <div className="space-y-2 text-xs">
              <div className="bg-white p-2 rounded">Google Embeddings</div>
              <div className="bg-white p-2 rounded">ChromaDB Store</div>
              <div className="bg-white p-2 rounded">MMR Algorithm</div>
              <div className="bg-blue-200 p-2 rounded text-center font-semibold">
                k=20 results
              </div>
            </div>
          </div>

          <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
            <h3 className="font-bold text-purple-800 mb-2 text-center">BM25 Search</h3>
            <div className="space-y-2 text-xs">
              <div className="bg-white p-2 rounded">Keyword Matching</div>
              <div className="bg-white p-2 rounded">TF-IDF Scoring</div>
              <div className="bg-white p-2 rounded">Thai Text Optimized</div>
              <div className="bg-purple-200 p-2 rounded text-center font-semibold">
                k=20 results
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Ensemble */}
        <div className="bg-indigo-50 border-2 border-indigo-300 rounded-lg p-4">
          <h3 className="font-bold text-indigo-800 mb-3 text-center">Ensemble Retriever</h3>
          <div className="flex justify-center gap-4 text-sm">
            <div className="bg-white px-4 py-2 rounded shadow-sm">
              Vector: 60% weight
            </div>
            <div className="text-2xl">+</div>
            <div className="bg-white px-4 py-2 rounded shadow-sm">
              BM25: 40% weight
            </div>
          </div>
          <div className="mt-3 text-center bg-indigo-200 p-2 rounded font-semibold">
            Combined Top 20 candidates
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Reranking */}
        <div className="bg-pink-50 border-2 border-pink-300 rounded-lg p-4">
          <h3 className="font-bold text-pink-800 mb-3 text-center">Cohere Reranking</h3>
          <div className="space-y-2 text-sm">
            <div className="bg-white p-2 rounded">Model: rerank-multilingual-v3.0</div>
            <div className="bg-white p-2 rounded">Cross-encoder architecture</div>
            <div className="bg-white p-2 rounded">Query-document relevance scoring</div>
            <div className="bg-pink-200 p-3 rounded text-center font-semibold">
              Final Top 10 most relevant chunks
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Context Formation */}
        <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-4">
          <h3 className="font-bold text-orange-800 mb-3 text-center">Context Formation</h3>
          <div className="space-y-2 text-sm">
            <div className="bg-white p-2 rounded">Format top 10 chunks</div>
            <div className="bg-white p-2 rounded">Clean formatting artifacts</div>
            <div className="bg-white p-2 rounded">Remove citations</div>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="text-2xl text-gray-400">↓</div>
        </div>

        {/* Final Output */}
        <div className="text-center">
          <div className="inline-block bg-green-100 border-2 border-green-400 rounded-lg px-6 py-3">
            <div className="font-bold text-green-800">Context for LLM</div>
            <div className="text-xs text-green-600 mt-2 text-left max-w-md">
              "โครงงานรายวิชา Microcontroller เป็นโครงงานกลุ่ม แบ่งกลุ่ม 5-6 คน
              คิดเป็น 20 คะแนน มีการประเมินจาก..."
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemDiagrams;