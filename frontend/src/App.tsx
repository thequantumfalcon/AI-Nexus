import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { QuantumTab } from './components/QuantumTab';
import { DashboardTab } from './components/DashboardTab';

function App() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">AI-Nexus</h1>
      <Tabs defaultValue="quantum" className="w-full">
        <TabsList>
          <TabsTrigger value="quantum">Quantum</TabsTrigger>
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="nlp">NLP</TabsTrigger>
        </TabsList>
        <TabsContent value="quantum">
          <QuantumTab />
        </TabsContent>
        <TabsContent value="dashboard">
          <DashboardTab />
        </TabsContent>
        <TabsContent value="nlp">
          <div>NLP RAG interface</div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default App;