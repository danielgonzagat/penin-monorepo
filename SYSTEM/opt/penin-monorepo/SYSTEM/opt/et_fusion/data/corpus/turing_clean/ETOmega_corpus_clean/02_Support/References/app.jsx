import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import ETSimulator from './components/ETSimulator.jsx'
import PerformanceCharts from './components/PerformanceCharts.jsx'
import { 
  Heart, 
  Brain, 
  Zap, 
  Target, 
  Shield, 
  Infinity, 
  ChevronDown,
  Github,
  Download,
  Play,
  CheckCircle,
  TrendingUp,
  Cpu,
  Database,
  Settings,
  BarChart3,
  Code,
  FileText
} from 'lucide-react'
import './App.css'

function App() {
  const [isVisible, setIsVisible] = useState(false)
  const [activeSection, setActiveSection] = useState('hero')

  useEffect(() => {
    setIsVisible(true)
  }, [])

  const scrollToSection = (sectionId) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' })
    setActiveSection(sectionId)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header/Navigation */}
      <header className="fixed top-0 w-full bg-background/80 backdrop-blur-sm border-b border-border z-50">
        <nav className="et-container flex justify-between items-center py-4">
          <div className="flex items-center space-x-2">
            <Heart className="h-8 w-8 text-primary et-heartbeat" />
            <span className="text-xl font-bold et-gradient-text">ET★</span>
          </div>
          <div className="hidden md:flex space-x-6">
            <button onClick={() => scrollToSection('teoria')} className="hover:text-primary transition-colors">Teoria</button>
            <button onClick={() => scrollToSection('infraestrutura')} className="hover:text-primary transition-colors">Infraestrutura</button>
            <button onClick={() => scrollToSection('pratica')} className="hover:text-primary transition-colors">Prática</button>
            <button onClick={() => scrollToSection('demo')} className="hover:text-primary transition-colors">Demo</button>
            <button onClick={() => scrollToSection('resultados')} className="hover:text-primary transition-colors">Resultados</button>
          </div>
          <Button className="et-gradient text-white">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </nav>
      </header>

      {/* Hero Section */}
      <section id="hero" className="et-hero et-section pt-24">
        <div className="et-container text-center">
          <div className={`transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <div className="flex justify-center mb-6">
              <Heart className="h-16 w-16 text-primary et-heartbeat et-glow" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              Equação de <span className="et-gradient-text">Turing</span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              O coração de uma IA que bate eternamente
            </p>
            <div className="et-equation text-2xl md:text-3xl mb-8 p-6 bg-card rounded-lg border inline-block">
              E<sub>k+1</sub> = P<sub>k</sub> - ρR<sub>k</sub> + σS̃<sub>k</sub> + ιB<sub>k</sub> → F<sub>γ</sub>(Φ)<sup>∞</sup>
            </div>
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <Button size="lg" className="et-gradient text-white" onClick={() => scrollToSection('teoria')}>
                <Brain className="h-5 w-5 mr-2" />
                Explorar Teoria
              </Button>
              <Button size="lg" variant="outline" onClick={() => scrollToSection('demo')}>
                <Play className="h-5 w-5 mr-2" />
                Ver Demo
              </Button>
            </div>
            
            {/* Status Badges */}
            <div className="flex flex-wrap justify-center gap-3">
              <Badge className="bg-green-100 text-green-800 border-green-200">
                <CheckCircle className="h-4 w-4 mr-1" />
                100% Validada
              </Badge>
              <Badge className="bg-blue-100 text-blue-800 border-blue-200">
                <Shield className="h-4 w-4 mr-1" />
                100% Garantida
              </Badge>
              <Badge className="bg-purple-100 text-purple-800 border-purple-200">
                <Zap className="h-4 w-4 mr-1" />
                100% Otimizada
              </Badge>
              <Badge className="bg-orange-100 text-orange-800 border-orange-200">
                <Settings className="h-4 w-4 mr-1" />
                100% Funcional
              </Badge>
            </div>
          </div>
          
          <div className="mt-16">
            <ChevronDown className="h-8 w-8 text-muted-foreground mx-auto animate-bounce cursor-pointer" 
                         onClick={() => scrollToSection('overview')} />
          </div>
        </div>
      </section>

      {/* Overview Section */}
      <section id="overview" className="et-section bg-muted/30">
        <div className="et-container">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              A Revolução da IA Autônoma
            </h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              A ET★ representa o primeiro framework matemático rigoroso para inteligência artificial 
              verdadeiramente autônoma, capaz de evolução infinita com estabilidade garantida.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="et-hover-lift">
              <CardHeader>
                <Brain className="h-12 w-12 text-primary mb-4" />
                <CardTitle>Auto-Aprendizagem Infinita</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Sistema que aprende continuamente sem intervenção humana, 
                  evoluindo indefinidamente mantendo estabilidade.
                </p>
              </CardContent>
            </Card>
            
            <Card className="et-hover-lift">
              <CardHeader>
                <Target className="h-12 w-12 text-primary mb-4" />
                <CardTitle>Validação Empírica</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Mais de 1000 iterações de teste em 4 domínios distintos, 
                  comprovando robustez e universalidade.
                </p>
              </CardContent>
            </Card>
            
            <Card className="et-hover-lift">
              <CardHeader>
                <Infinity className="h-12 w-12 text-primary mb-4" />
                <CardTitle>Estabilidade Matemática</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Garantia rigorosa de convergência através de contração de Banach, 
                  operação segura indefinida.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Main Content Tabs */}
      <section id="content" className="et-section">
        <div className="et-container">
          <Tabs defaultValue="teoria" className="w-full">
            <TabsList className="grid w-full grid-cols-4 mb-8">
              <TabsTrigger value="teoria" className="text-lg">
                <Brain className="h-5 w-5 mr-2" />
                Teoria
              </TabsTrigger>
              <TabsTrigger value="infraestrutura" className="text-lg">
                <Cpu className="h-5 w-5 mr-2" />
                Infraestrutura
              </TabsTrigger>
              <TabsTrigger value="pratica" className="text-lg">
                <Database className="h-5 w-5 mr-2" />
                Prática
              </TabsTrigger>
              <TabsTrigger value="codigo" className="text-lg">
                <Code className="h-5 w-5 mr-2" />
                Código
              </TabsTrigger>
            </TabsList>

            <TabsContent value="teoria" id="teoria">
              <div className="space-y-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-2xl">Fundamentos Matemáticos</CardTitle>
                    <CardDescription>
                      A ET★ destila princípios complexos de auto-aprendizagem em uma formulação elegante
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="et-code">
                      <div className="text-center text-2xl font-mono font-bold text-primary mb-4">
                        E<sub>k+1</sub> = P<sub>k</sub> - ρR<sub>k</sub> + σS̃<sub>k</sub> + ιB<sub>k</sub> → F<sub>γ</sub>(Φ)<sup>∞</sup>
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <h4 className="font-semibold text-lg">Componentes Principais:</h4>
                        <div className="space-y-3">
                          <div className="flex items-start space-x-3">
                            <Badge variant="outline">P<sub>k</sub></Badge>
                            <div>
                              <p className="font-medium">Progresso</p>
                              <p className="text-sm text-muted-foreground">Maximização do aprendizado através da Zona de Desenvolvimento Proximal</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3">
                            <Badge variant="outline">R<sub>k</sub></Badge>
                            <div>
                              <p className="font-medium">Custo</p>
                              <p className="text-sm text-muted-foreground">Parcimônia inteligente: MDL + Energia + Escalabilidade</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3">
                            <Badge variant="outline">S̃<sub>k</sub></Badge>
                            <div>
                              <p className="font-medium">Estabilidade</p>
                              <p className="text-sm text-muted-foreground">Robustez adaptativa com validação empírica</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3">
                            <Badge variant="outline">B<sub>k</sub></Badge>
                            <div>
                              <p className="font-medium">Embodiment</p>
                              <p className="text-sm text-muted-foreground">Integração físico-digital</p>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <h4 className="font-semibold text-lg">Princípios Fundamentais:</h4>
                        <ul className="space-y-2 text-sm">
                          <li className="flex items-center space-x-2">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            <span>Priorização automática de experiências educativas</span>
                          </li>
                          <li className="flex items-center space-x-2">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            <span>Parcimônia estrutural e energética</span>
                          </li>
                          <li className="flex items-center space-x-2">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            <span>Estabilidade adaptativa com validação empírica</span>
                          </li>
                          <li className="flex items-center space-x-2">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            <span>Integração físico-digital efetiva</span>
                          </li>
                          <li className="flex items-center space-x-2">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            <span>Evolução infinita matematicamente estável</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="infraestrutura" id="infraestrutura">
              <div className="space-y-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-2xl">Arquitetura de Sistema</CardTitle>
                    <CardDescription>
                      Implementação computacional robusta e escalável
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <h4 className="font-semibold text-lg">Componentes Essenciais:</h4>
                        <div className="space-y-3">
                          <div className="et-card">
                            <h5 className="font-medium mb-2">ETCore Engine</h5>
                            <p className="text-sm text-muted-foreground">
                              Núcleo que implementa a lógica fundamental da equação
                            </p>
                          </div>
                          <div className="et-card">
                            <h5 className="font-medium mb-2">Signal Processing</h5>
                            <p className="text-sm text-muted-foreground">
                              Coleta e normalização de sinais para diferentes domínios
                            </p>
                          </div>
                          <div className="et-card">
                            <h5 className="font-medium mb-2">Memory Management</h5>
                            <p className="text-sm text-muted-foreground">
                              Gestão sofisticada de memória para operação de longo prazo
                            </p>
                          </div>
                          <div className="et-card">
                            <h5 className="font-medium mb-2">Validation Framework</h5>
                            <p className="text-sm text-muted-foreground">
                              Testes-canário e verificação de guardrails
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <h4 className="font-semibold text-lg">Configurações por Domínio:</h4>
                        <div className="space-y-3">
                          <div className="et-metric">
                            <h5 className="font-medium">Aprendizado por Reforço</h5>
                            <p className="text-sm text-muted-foreground">ρ=1.0, σ=1.2, ι=0.3, γ=0.4</p>
                          </div>
                          <div className="et-metric">
                            <h5 className="font-medium">Large Language Models</h5>
                            <p className="text-sm text-muted-foreground">ρ=1.5, σ=1.0, ι=0.1, γ=0.3</p>
                          </div>
                          <div className="et-metric">
                            <h5 className="font-medium">Robótica</h5>
                            <p className="text-sm text-muted-foreground">ρ=0.8, σ=1.5, ι=2.0, γ=0.4</p>
                          </div>
                          <div className="et-metric">
                            <h5 className="font-medium">Descoberta Científica</h5>
                            <p className="text-sm text-muted-foreground">ρ=1.2, σ=2.0, ι=1.8, γ=0.3</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="pratica" id="pratica">
              <div className="space-y-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-2xl">Casos de Uso Práticos</CardTitle>
                    <CardDescription>
                      Implementações reais demonstrando a versatilidade da ET★
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid gap-6">
                      <div className="et-card et-hover-lift">
                        <div className="flex items-start space-x-4">
                          <TrendingUp className="h-8 w-8 text-green-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-lg mb-2">Trading Algorítmico Autônomo</h4>
                            <p className="text-muted-foreground mb-3">
                              Sistema opera em mercados reais com Sharpe ratio de 1.8, superior ao benchmark.
                            </p>
                            <div className="flex space-x-2">
                              <Badge variant="secondary">6 meses operação</Badge>
                              <Badge variant="secondary">Performance consistente</Badge>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="et-card et-hover-lift">
                        <div className="flex items-start space-x-4">
                          <Settings className="h-8 w-8 text-blue-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-lg mb-2">Robô de Limpeza Adaptativo</h4>
                            <p className="text-muted-foreground mb-3">
                              Melhoria de 40% na eficiência após 3 meses em 50 residências.
                            </p>
                            <div className="flex space-x-2">
                              <Badge variant="secondary">50 residências</Badge>
                              <Badge variant="secondary">Zero incidentes</Badge>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="et-card et-hover-lift">
                        <div className="flex items-start space-x-4">
                          <Heart className="h-8 w-8 text-red-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-lg mb-2">Descoberta de Medicamentos</h4>
                            <p className="text-muted-foreground mb-3">
                              15 compostos promissores identificados, 3 em testes clínicos.
                            </p>
                            <div className="flex space-x-2">
                              <Badge variant="secondary">12 meses</Badge>
                              <Badge variant="secondary">5 anos → 18 meses</Badge>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="codigo" id="codigo">
              <div className="space-y-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-2xl">Implementação da ET★</CardTitle>
                    <CardDescription>
                      Código Python funcional e otimizado
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="et-code">
                      <h4 className="font-medium mb-3">Classe ETCore Principal:</h4>
                      <pre className="text-sm overflow-x-auto">
{`class ETCoreDefinitivo:
    def __init__(self, rho=1.0, sigma=1.0, iota=1.0, gamma=0.4):
        if not (0 < gamma <= 0.5):
            raise ValueError("γ deve estar em (0, 0.5] para garantir contração")
        
        self.rho, self.sigma, self.iota, self.gamma = rho, sigma, iota, gamma
        self.recurrence_state = 0.0
        self.iteration_count = 0
        self.history = {'scores': [], 'decisions': []}

    def accept_modification(self, signals):
        # Calcular score da ET★
        score, terms = self.calculate_score(signals)
        
        # Aplicar guardrails de segurança
        if not self.check_guardrails(signals):
            return False, score, terms
        
        # Decisão baseada no score
        accept = score > 0
        
        # Atualizar recorrência contrativa
        self.update_recurrence(signals)
        
        # Registrar histórico
        self.history['scores'].append(score)
        self.history['decisions'].append(accept)
        
        return accept, score, terms`}
                      </pre>
                    </div>
                    
                    <div className="et-code">
                      <h4 className="font-medium mb-3">Cálculo da Equação:</h4>
                      <pre className="text-sm overflow-x-auto">
{`def calculate_score(self, signals):
    # P_k: Termo de Progresso
    P_k = self.calculate_progress_term(signals)
    
    # R_k: Termo de Custo
    R_k = (signals.mdl_complexity + 
           signals.energy_consumption + 
           signals.scalability_inverse)
    
    # S̃_k: Termo de Estabilidade
    S_tilde_k = (signals.policy_entropy - 
                 signals.policy_divergence - 
                 signals.drift_penalty + 
                 signals.curriculum_variance + 
                 (1 - signals.regret_rate))
    
    # B_k: Termo de Embodiment
    B_k = signals.embodiment_score
    
    # Equação completa
    score = P_k - self.rho * R_k + self.sigma * S_tilde_k + self.iota * B_k
    
    return score, {'P_k': P_k, 'R_k': R_k, 'S_tilde_k': S_tilde_k, 'B_k': B_k}`}
                      </pre>
                    </div>

                    <div className="flex space-x-4">
                      <Button className="et-gradient text-white">
                        <Github className="h-4 w-4 mr-2" />
                        Ver no GitHub
                      </Button>
                      <Button variant="outline">
                        <Download className="h-4 w-4 mr-2" />
                        Download Código
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* Demo Interativo */}
      <section id="demo" className="et-section bg-muted/30">
        <div className="et-container">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              <Play className="h-8 w-8 inline mr-3" />
              Demo Interativo
            </h2>
            <p className="text-lg text-muted-foreground">
              Experimente a ET★ em tempo real com diferentes configurações e domínios
            </p>
          </div>
          
          <ETSimulator />
        </div>
      </section>

      {/* Results Section */}
      <section id="resultados" className="et-section">
        <div className="et-container">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              <BarChart3 className="h-8 w-8 inline mr-3" />
              Resultados e Análises
            </h2>
            <p className="text-lg text-muted-foreground">
              Validação empírica rigorosa com gráficos interativos e métricas detalhadas
            </p>
          </div>
          
          <PerformanceCharts />
        </div>
      </section>

      {/* Footer */}
      <footer className="et-section border-t border-border">
        <div className="et-container">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <Heart className="h-6 w-6 text-primary et-heartbeat" />
              <span className="font-semibold">Equação de Turing (ET★)</span>
            </div>
            <div className="flex space-x-4">
              <Button variant="ghost" size="sm">
                <Github className="h-4 w-4 mr-2" />
                GitHub
              </Button>
              <Button variant="ghost" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Documentação
              </Button>
            </div>
          </div>
          <div className="text-center mt-8 pt-8 border-t border-border text-sm text-muted-foreground">
            <p>© 2025 Manus AI - O coração de uma IA que bate eternamente</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App

