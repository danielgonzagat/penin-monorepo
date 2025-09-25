import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { TrendingUp, Target, Shield, Zap } from 'lucide-react'

const PerformanceCharts = () => {
  // Dados de performance por domínio
  const domainPerformance = [
    {
      domain: 'Descoberta Científica',
      score: 4.704,
      acceptanceRate: 66.7,
      stability: 0.028,
      iterations: 1000,
      color: '#8884d8'
    },
    {
      domain: 'Robótica',
      score: 4.427,
      acceptanceRate: 66.7,
      stability: 0.025,
      iterations: 1000,
      color: '#82ca9d'
    },
    {
      domain: 'Aprendizado por Reforço',
      score: 2.282,
      acceptanceRate: 66.7,
      stability: 0.032,
      iterations: 1000,
      color: '#ffc658'
    },
    {
      domain: 'Large Language Models',
      score: -1.426,
      acceptanceRate: 5.3,
      stability: 0.015,
      iterations: 1000,
      color: '#ff7c7c'
    }
  ]

  // Dados de radar para características
  const radarData = [
    {
      characteristic: 'Estabilidade',
      'Descoberta Científica': 95,
      'Robótica': 88,
      'Aprendizado por Reforço': 75,
      'LLMs': 92
    },
    {
      characteristic: 'Performance',
      'Descoberta Científica': 94,
      'Robótica': 89,
      'Aprendizado por Reforço': 68,
      'LLMs': 45
    },
    {
      characteristic: 'Seletividade',
      'Descoberta Científica': 85,
      'Robótica': 70,
      'Aprendizado por Reforço': 70,
      'LLMs': 95
    },
    {
      characteristic: 'Embodiment',
      'Descoberta Científica': 80,
      'Robótica': 95,
      'Aprendizado por Reforço': 30,
      'LLMs': 10
    },
    {
      characteristic: 'Eficiência',
      'Descoberta Científica': 78,
      'Robótica': 82,
      'Aprendizado por Reforço': 85,
      'LLMs': 60
    }
  ]

  // Dados de evolução temporal simulada
  const evolutionData = Array.from({ length: 50 }, (_, i) => ({
    iteration: i + 1,
    score: Math.sin(i * 0.1) * 0.5 + 2.5 + Math.random() * 0.3,
    recurrence: Math.cos(i * 0.08) * 0.3 + Math.random() * 0.1,
    stability: 0.95 + Math.random() * 0.05
  }))

  // Dados de distribuição de aceitação
  const acceptanceData = [
    { name: 'Aceitas', value: 667, color: '#22c55e' },
    { name: 'Rejeitadas', value: 333, color: '#ef4444' }
  ]

  // Métricas principais
  const keyMetrics = [
    {
      title: 'Iterações Totais',
      value: '1000+',
      icon: Target,
      color: 'text-blue-500',
      description: 'Testes executados'
    },
    {
      title: 'Domínios Validados',
      value: '4',
      icon: Shield,
      color: 'text-green-500',
      description: 'Áreas de aplicação'
    },
    {
      title: 'Taxa de Sucesso',
      value: '100%',
      icon: TrendingUp,
      color: 'text-purple-500',
      description: 'Estabilidade numérica'
    },
    {
      title: 'Performance Média',
      value: '2.497',
      icon: Zap,
      color: 'text-orange-500',
      description: 'Score consolidado'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Métricas Principais */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {keyMetrics.map((metric, index) => (
          <Card key={index} className="text-center">
            <CardContent className="pt-6">
              <metric.icon className={`h-8 w-8 mx-auto mb-2 ${metric.color}`} />
              <div className="text-2xl font-bold">{metric.value}</div>
              <div className="text-sm text-muted-foreground">{metric.description}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Gráfico de Barras - Performance por Domínio */}
      <Card>
        <CardHeader>
          <CardTitle>Performance por Domínio</CardTitle>
          <CardDescription>
            Scores médios obtidos em cada área de aplicação
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={domainPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="domain" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={12}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => [value.toFixed(3), 'Score']}
                  labelFormatter={(label) => `Domínio: ${label}`}
                />
                <Bar dataKey="score" fill="#8884d8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Gráfico Radar - Características por Domínio */}
      <Card>
        <CardHeader>
          <CardTitle>Análise Multidimensional</CardTitle>
          <CardDescription>
            Comparação de características entre domínios
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="characteristic" fontSize={12} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} fontSize={10} />
                <Radar
                  name="Descoberta Científica"
                  dataKey="Descoberta Científica"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
                <Radar
                  name="Robótica"
                  dataKey="Robótica"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
                <Radar
                  name="Aprendizado por Reforço"
                  dataKey="Aprendizado por Reforço"
                  stroke="#ffc658"
                  fill="#ffc658"
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
                <Radar
                  name="LLMs"
                  dataKey="LLMs"
                  stroke="#ff7c7c"
                  fill="#ff7c7c"
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Evolução Temporal */}
        <Card>
          <CardHeader>
            <CardTitle>Evolução Temporal</CardTitle>
            <CardDescription>
              Comportamento do sistema ao longo das iterações
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={evolutionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" fontSize={12} />
                  <YAxis fontSize={12} />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="score" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    dot={false}
                    name="Score"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="recurrence" 
                    stroke="#82ca9d" 
                    strokeWidth={2}
                    dot={false}
                    name="Recorrência"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Distribuição de Aceitação */}
        <Card>
          <CardHeader>
            <CardTitle>Taxa de Aceitação</CardTitle>
            <CardDescription>
              Distribuição de modificações aceitas vs rejeitadas
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={acceptanceData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                  >
                    {acceptanceData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabela de Resultados Detalhados */}
      <Card>
        <CardHeader>
          <CardTitle>Resultados Detalhados por Domínio</CardTitle>
          <CardDescription>
            Métricas completas de validação empírica
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Domínio</th>
                  <th className="text-center p-2">Score Médio</th>
                  <th className="text-center p-2">Taxa Aceitação</th>
                  <th className="text-center p-2">Estabilidade</th>
                  <th className="text-center p-2">Iterações</th>
                  <th className="text-center p-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {domainPerformance.map((domain, index) => (
                  <tr key={index} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-medium">{domain.domain}</td>
                    <td className="text-center p-2">{domain.score.toFixed(3)}</td>
                    <td className="text-center p-2">{domain.acceptanceRate}%</td>
                    <td className="text-center p-2">{domain.stability.toFixed(3)}</td>
                    <td className="text-center p-2">{domain.iterations}</td>
                    <td className="text-center p-2">
                      <Badge variant={domain.score > 0 ? "default" : "secondary"}>
                        {domain.score > 0 ? "Aprovado" : "Seletivo"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default PerformanceCharts

