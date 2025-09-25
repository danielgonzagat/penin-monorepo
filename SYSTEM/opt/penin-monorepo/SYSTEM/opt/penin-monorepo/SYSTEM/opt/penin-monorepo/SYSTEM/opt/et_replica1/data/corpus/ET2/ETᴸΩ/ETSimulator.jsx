import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Slider } from '@/components/ui/slider.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Play, Pause, RotateCcw, Settings } from 'lucide-react'

const ETSimulator = () => {
  const [isRunning, setIsRunning] = useState(false)
  const [iteration, setIteration] = useState(0)
  const [parameters, setParameters] = useState({
    rho: 1.0,
    sigma: 1.2,
    iota: 0.3,
    gamma: 0.4
  })
  const [currentScore, setCurrentScore] = useState(0)
  const [recurrenceState, setRecurrenceState] = useState(0)
  const [history, setHistory] = useState([])
  const [domain, setDomain] = useState('rl')

  const domainConfigs = {
    rl: { rho: 1.0, sigma: 1.2, iota: 0.3, gamma: 0.4, name: 'Aprendizado por Reforço' },
    llm: { rho: 1.5, sigma: 1.0, iota: 0.1, gamma: 0.3, name: 'Large Language Models' },
    robotics: { rho: 0.8, sigma: 1.5, iota: 2.0, gamma: 0.4, name: 'Robótica' },
    science: { rho: 1.2, sigma: 2.0, iota: 1.8, gamma: 0.3, name: 'Descoberta Científica' }
  }

  // Simulação da ET★
  const simulateETStep = () => {
    // Gerar sinais simulados
    const learningProgress = Math.random() * 0.8 + 0.1
    const taskDifficulty = Math.random() * 0.9 + 0.1
    const mdlComplexity = Math.random() * 2 + 0.5
    const energyConsumption = Math.random() * 1.5 + 0.2
    const scalabilityInverse = Math.random() * 0.8 + 0.2
    const policyEntropy = Math.random() * 0.9 + 0.1
    const policyDivergence = Math.random() * 0.3
    const driftPenalty = Math.random() * 0.2
    const curriculumVariance = Math.random() * 0.5 + 0.1
    const regretRate = Math.random() * 0.15
    const embodimentScore = Math.random() * 0.9 + 0.1

    // Calcular termos da equação
    const P_k = learningProgress * taskDifficulty * 1.5
    const R_k = mdlComplexity + energyConsumption + scalabilityInverse
    const S_tilde_k = policyEntropy - policyDivergence - driftPenalty + curriculumVariance + (1 - regretRate)
    const B_k = embodimentScore

    // Aplicar parâmetros
    const score = P_k - parameters.rho * R_k + parameters.sigma * S_tilde_k + parameters.iota * B_k

    // Atualizar recorrência contrativa
    const phi = Math.tanh(score * 0.1)
    const newRecurrenceState = (1 - parameters.gamma) * recurrenceState + parameters.gamma * phi
    
    return {
      score,
      recurrenceState: Math.max(-1, Math.min(1, newRecurrenceState)),
      terms: { P_k, R_k, S_tilde_k, B_k },
      accepted: score > 0 && Math.abs(newRecurrenceState) < 0.9
    }
  }

  useEffect(() => {
    let interval
    if (isRunning) {
      interval = setInterval(() => {
        const result = simulateETStep()
        setCurrentScore(result.score)
        setRecurrenceState(result.recurrenceState)
        setIteration(prev => prev + 1)
        
        setHistory(prev => {
          const newHistory = [...prev, {
            iteration: iteration + 1,
            score: result.score,
            recurrence: result.recurrenceState,
            accepted: result.accepted,
            ...result.terms
          }].slice(-50) // Manter apenas últimos 50 pontos
          return newHistory
        })
      }, 500)
    }
    return () => clearInterval(interval)
  }, [isRunning, iteration, recurrenceState, parameters])

  const reset = () => {
    setIsRunning(false)
    setIteration(0)
    setCurrentScore(0)
    setRecurrenceState(0)
    setHistory([])
  }

  const changeDomain = (newDomain) => {
    setDomain(newDomain)
    setParameters(domainConfigs[newDomain])
    reset()
  }

  const acceptanceRate = history.length > 0 ? 
    (history.filter(h => h.accepted).length / history.length * 100).toFixed(1) : 0

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Simulador ET★ Interativo</span>
          </CardTitle>
          <CardDescription>
            Experimente a Equação de Turing em tempo real com diferentes configurações
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Controles de Domínio */}
          <div>
            <h4 className="font-medium mb-3">Domínio de Aplicação:</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(domainConfigs).map(([key, config]) => (
                <Button
                  key={key}
                  variant={domain === key ? "default" : "outline"}
                  size="sm"
                  onClick={() => changeDomain(key)}
                  className="text-xs"
                >
                  {config.name}
                </Button>
              ))}
            </div>
          </div>

          {/* Controles de Simulação */}
          <div className="flex items-center space-x-4">
            <Button
              onClick={() => setIsRunning(!isRunning)}
              className={isRunning ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"}
            >
              {isRunning ? (
                <>
                  <Pause className="h-4 w-4 mr-2" />
                  Pausar
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Iniciar
                </>
              )}
            </Button>
            <Button variant="outline" onClick={reset}>
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset
            </Button>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Iteração:</span>
              <Badge variant="secondary">{iteration}</Badge>
            </div>
          </div>

          {/* Parâmetros */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium">Parâmetros da ET★:</h4>
              {Object.entries(parameters).map(([key, value]) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium">{key} (ρ, σ, ι, γ):</label>
                    <span className="text-sm">{value.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[value]}
                    onValueChange={(newValue) => 
                      setParameters(prev => ({ ...prev, [key]: newValue[0] }))
                    }
                    min={0.1}
                    max={key === 'gamma' ? 0.5 : 2.0}
                    step={0.1}
                    className="w-full"
                  />
                </div>
              ))}
            </div>

            <div className="space-y-4">
              <h4 className="font-medium">Status Atual:</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">Score Atual:</span>
                    <span className="text-sm font-medium">{currentScore.toFixed(3)}</span>
                  </div>
                  <Progress 
                    value={Math.max(0, Math.min(100, (currentScore + 5) * 10))} 
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">Estado Recorrência:</span>
                    <span className="text-sm font-medium">{recurrenceState.toFixed(3)}</span>
                  </div>
                  <Progress 
                    value={(recurrenceState + 1) * 50} 
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">Taxa de Aceitação:</span>
                    <span className="text-sm font-medium">{acceptanceRate}%</span>
                  </div>
                  <Progress 
                    value={acceptanceRate} 
                    className="h-2"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Gráfico de Performance */}
          {history.length > 0 && (
            <div>
              <h4 className="font-medium mb-3">Evolução do Score:</h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="iteration" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="score" 
                      stroke="#8884d8" 
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="recurrence" 
                      stroke="#82ca9d" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Equação Atual */}
          <div className="et-code text-center">
            <div className="text-lg font-mono font-bold text-primary">
              E<sub>k+1</sub> = P<sub>k</sub> - {parameters.rho}R<sub>k</sub> + {parameters.sigma}S̃<sub>k</sub> + {parameters.iota}B<sub>k</sub> → F<sub>{parameters.gamma}</sub>(Φ)<sup>∞</sup>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ETSimulator

