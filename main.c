#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// Parâmetros de configuração do PSO
#define NUM_PARTICULAS 500
#define NUM_ITERACOES 1000               //é interessante testar com mais iterações, como 100k
#define NUM_ATIVOS 30
#define C1 2.0
#define C2 2.0
#define W 0.7
#define MIN_ATIVOS 5
#define MAX_ATIVOS_SELECIONADOS 10
#define MAX_ALOCACAO 0.3
#define MAX_LINHAS 5000

#define NUM_THREADS 16

// Estrutura para representar uma partícula
typedef struct {
    double posicao[NUM_ATIVOS];
    double velocidade[NUM_ATIVOS];
    double melhor_posicao[NUM_ATIVOS];
    double fitness_atual;
    double melhor_fitness;
    char padding[64]; // Padding para evitar false sharing
} Particula;

// Função para gerar dados sintéticos de preços dos ativos
void gerar_precos_aleatorios(double precos[NUM_ATIVOS][MAX_LINHAS]) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_ATIVOS; i++) {
        for (int j = 0; j < MAX_LINHAS; j++) {
            // Preços aleatórios entre 100 e 200
            precos[i][j] = 100 + (rand() % 101);
        }
    }
}

// Função para calcular o risco (volatilidade) de cada ativo com base nos preços
void calcular_riscos(double precos[NUM_ATIVOS][MAX_LINHAS], double riscos[NUM_ATIVOS]) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_ATIVOS; i++) {
        double somatorio = 0.0;
        for (int j = 1; j < MAX_LINHAS; j++) {
            double retorno = (precos[i][j] - precos[i][j-1]) / precos[i][j-1];
            somatorio += retorno * retorno;
        }
        double risco = sqrt(somatorio / (MAX_LINHAS - 1));
        risco *= 1.0 + (double)rand() / RAND_MAX;
        riscos[i] = fmin(risco, 10.0);
    }
}

// Função para calcular o fitness (função objetivo)
// Retorna -INFINITY se alguma restrição for violada
double calcular_fitness(double* posicao, double riscos[NUM_ATIVOS]) {
    double risco_portfolio = 0.0;
    double soma_pesos = 0.0;
    int ativos_selecionados = 0;

    for (int i = 0; i < NUM_ATIVOS; i++) {
        if (posicao[i] > 0.01) {
            risco_portfolio += posicao[i] * riscos[i];
            soma_pesos += posicao[i];
            ativos_selecionados++;

            if (posicao[i] > MAX_ALOCACAO) {
                return -INFINITY;
            }
        }
    }

    if (ativos_selecionados < MIN_ATIVOS || ativos_selecionados > MAX_ATIVOS_SELECIONADOS || fabs(soma_pesos - 1.0) > 1e-6) {
        return -INFINITY;
    }

    return -risco_portfolio;
}

// Inicializar partícula
void inicializar_particula(Particula* particula, double riscos[NUM_ATIVOS]) {
    double soma = 0.0;

    #pragma omp parallel for reduction(+:soma)
    for (int i = 0; i < NUM_ATIVOS; i++) {
        // rand() pode causar problemas aqui (e em outros lugares paralelizados) por não ser thread safe
        particula->posicao[i] = (double)rand() / RAND_MAX;
        particula->velocidade[i] = (((double)rand() / RAND_MAX) - 0.5) * 0.2;
        soma += particula->posicao[i];
    }

    // Verifica que a soma não seja zero (raríssimo, mas para segurança)
    if (fabs(soma) < 1e-12)
        soma = 1.0;

    for (int i = 0; i < NUM_ATIVOS; i++) {
        particula->posicao[i] /= soma;
        particula->melhor_posicao[i] = particula->posicao[i];
    }

    particula->fitness_atual = calcular_fitness(particula->posicao, riscos);
    particula->melhor_fitness = particula->fitness_atual;
}

// Atualizar velocidade e posição da partícula
void atualizar_particula(Particula* particula, double* melhor_portfolio_global, double riscos[NUM_ATIVOS]) {
    // Atualiza os ativos (iteração sobre NUM_ATIVOS: apenas 30 iterações)
    for (int i = 0; i < NUM_ATIVOS; i++) {
        double r1 = (double)rand() / RAND_MAX;
        double r2 = (double)rand() / RAND_MAX;

        particula->velocidade[i] = W * particula->velocidade[i] +
                                   C1 * r1 * (particula->melhor_posicao[i] - particula->posicao[i]) +
                                   C2 * r2 * (melhor_portfolio_global[i] - particula->posicao[i]);

        particula->posicao[i] += particula->velocidade[i];
        // Limita os valores para garantir que fiquem no intervalo [0,1]
        if (particula->posicao[i] < 0.0)
            particula->posicao[i] = 0.0;
        if (particula->posicao[i] > 1.0)
            particula->posicao[i] = 1.0;
    }

    // Recalcula a soma para normalizar as posições
    double soma = 0.0;
    for (int i = 0; i < NUM_ATIVOS; i++) {
        soma += particula->posicao[i];
    }
    if (fabs(soma) < 1e-12)
        soma = 1.0;
    for (int i = 0; i < NUM_ATIVOS; i++) {
        particula->posicao[i] /= soma;
    }

    particula->fitness_atual = calcular_fitness(particula->posicao, riscos);

    if (particula->fitness_atual > particula->melhor_fitness) {
        particula->melhor_fitness = particula->fitness_atual;
        // Atualiza a melhor posição
        for (int i = 0; i < NUM_ATIVOS; i++) {
            particula->melhor_posicao[i] = particula->posicao[i];
        }
    }
}

// Função principal de otimização por enxame de partículas
void otimizar_portfolio() {
    srand(time(NULL));
    double riscos[NUM_ATIVOS];
    double precos[NUM_ATIVOS][MAX_LINHAS];

    gerar_precos_aleatorios(precos);
    calcular_riscos(precos, riscos);

    // Exibe os riscos dos ativos
    /*printf("Riscos dos Ativos:\n");
    for (int i = 0; i < NUM_ATIVOS; i++) {
        printf("Ativo %d: %.4f\n", i, riscos[i]);
    }*/

    Particula enxame[NUM_PARTICULAS];
    double melhor_portfolio_global[NUM_ATIVOS] = {0};
    double melhor_fitness_global = -INFINITY;

    // Inicializa as partículas em paralelo
    #pragma omp parallel for
    for (int i = 0; i < NUM_PARTICULAS; i++) {
        inicializar_particula(&enxame[i], riscos);
        #pragma omp critical
        {
            if (enxame[i].fitness_atual > melhor_fitness_global) {
                melhor_fitness_global = enxame[i].fitness_atual;
                for (int j = 0; j < NUM_ATIVOS; j++) {
                    melhor_portfolio_global[j] = enxame[i].posicao[j];
                }
            }
        }
    }

    // Exibe os pesos iniciais de algumas partículas
    /*printf("\nPesos Iniciais das Partículas:\n");
    for (int i = 0; i < 5; i++) {
        printf("Partícula %d: ", i);
        for (int j = 0; j < NUM_ATIVOS; j++) {
            printf("%.4f ", enxame[i].posicao[j]);
        }
        printf("\n");
    }*/

    // Processo iterativo de otimização
    for (int iter = 0; iter < NUM_ITERACOES; iter++) {
        #pragma omp parallel
        {
            double fitness_local = -INFINITY;
            double portfolio_local[NUM_ATIVOS] = {0};

            #pragma omp for
            for (int i = 0; i < NUM_PARTICULAS; i++) {
                atualizar_particula(&enxame[i], melhor_portfolio_global, riscos);
                if (enxame[i].fitness_atual > fitness_local) {
                    fitness_local = enxame[i].fitness_atual;
                    for (int j = 0; j < NUM_ATIVOS; j++) {
                        portfolio_local[j] = enxame[i].posicao[j];
                    }
                }
            }

            #pragma omp critical
            {
                if (fitness_local > melhor_fitness_global) {
                    melhor_fitness_global = fitness_local;
                    for (int j = 0; j < NUM_ATIVOS; j++) {
                        melhor_portfolio_global[j] = portfolio_local[j];
                    }
                }
            }
        }

        // printf("Iteração %d: Melhor Fitness = %.4f\n", iter, melhor_fitness_global);
    }

    // Imprime os resultados
    printf("\nMelhor Portfolio Encontrado:\n");
    printf("Fitness: %.4f\n", melhor_fitness_global);
    printf("Ativos:\n");

    int ativos_selecionados = 0;
    for (int i = 0; i < NUM_ATIVOS; i++) {
        if (melhor_portfolio_global[i] > 0.01) {
            printf("Ativo %d: %.2f%%\n", i + 1, melhor_portfolio_global[i] * 100);
            ativos_selecionados++;
        }
    }
    printf("Total de ativos selecionados: %d\n", ativos_selecionados);
}

int main() {
    omp_set_num_threads(NUM_THREADS); // Define o número de threads

    double soma = 0;
    for (int i = 0; i < 40; i++) {
        double start = omp_get_wtime();
        otimizar_portfolio();
        double end = omp_get_wtime();
        soma += (end - start);
    }
    double media = soma / 40;
    printf("Tempo medio apos 40 testes com OpenMP: %f segundos\n", media);

}