
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <unordered_map>
#include <algorithm>
#include <iterator>

#include "Matrix_Class.h"
#include "Local_Decition_Tree_Class.h"


float f_max(float z) {
    return std::max(0.0f,z);
}

float delta_f_max(float a) {
    if (a > 0) return 1.0f;
    else return 0.0f;
}



//###########################################################################################################################
//###########################################################################################################################
//###########################################################################################################################
//###########################################################################################################################

std::vector<uint32_t> build_straight_lookup(int ranks) {
    std::vector<uint32_t> mask;
    for (int i = 0; i + 5 <= ranks; i++) {
        uint32_t m = 0;
        for (int j = i; j < i + 5; j++) m |= (1u << j);
        mask.push_back(m);
    }
    if (ranks == 13) {//ace low
        mask.push_back((1u << 12) | (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3));
    }
    return mask;
}




std::unordered_map<int, float> build_hand_lookup_table(int ranks, const std::vector <uint32_t>& strait_lookup, const long long P5, const long long P4, const long long P3, const long long P2, const long long P1) {
    long long n = 4 * ranks;
    int n_cards = n;
    std::vector<int> hands_scores;
    hands_scores.push_back(0);
    hands_scores.reserve(n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 120 + 1);
    for (int i = 0; i < n_cards; i++) {
        for (int j = i + 1; j < n_cards; j++) {
            for (int k = j + 1; k < n_cards; k++) {
                for (int l = k + 1; l < n_cards; l++) {
                    for (int o = l + 1; o < n_cards; o++) {
                        hands_scores.push_back(Decition_Node::hand_evaluation({ (float)i,(float)j,(float)k,(float)l,(float)o }, ranks, strait_lookup, P5, P4, P3, P2, P1));
                    }
                }
            }
        }
    }
    std::unordered_map<int, float> lookup;
    int count = hands_scores.size();
    std::sort(hands_scores.begin(), hands_scores.end());
    int start = 0;
    int score = hands_scores[0];
    int hand_score;
    for (int i = 0; i < count; i++) {
        hand_score = hands_scores[i];
        if (hand_score != score) {
            lookup[score] = (start + i) / (2.0f * count);
            start = i;
            score = hand_score;
        }
    }
    lookup[0] = 0.0f;
    lookup[score] = (start + count) / (2.0f * count);
    return lookup;
}

float linierisation(int state39, const std::unordered_map<int, float>& hand_lookup_table) {
    return hand_lookup_table.at(state39);
}

std::vector<float> gamstate_normalisation(float max_pot, float max_player_contribution, int ranks, int n_players, std::vector<float> gamestate_vec, int C, const std::unordered_map<int, float>& hand_lookup_table, int max_rounds) {
    gamestate_vec[0] /= max_rounds;
    gamestate_vec[1] /= max_pot;
    for (int i = 0; i < 2; i++) {
        gamestate_vec[2 + 5 * i] /= (ranks + 1);
    }
    for (int j = 0; j < 5; j++) {
        gamestate_vec[13 + 5 * j] /= (ranks + 1);
    }
    gamestate_vec[39] = linierisation(gamestate_vec[39], hand_lookup_table);
    for (int j = 0; j < n_players; j++) {
        gamestate_vec[40 + 3*n_players + j] /= max_player_contribution;
    }
    gamestate_vec[40 + 7 * n_players] /= C;
    gamestate_vec[41 + 7 * n_players] /= n_players;
    return gamestate_vec;
}

std::vector<float> regret_normalisation(float max_pot, std::vector<float> regret_vec) {
    for (int i = 0; i < 5; i++) {
        regret_vec[i] /= max_pot;
    }
    return regret_vec;
}


std::vector<float> hand(std::vector<float> deck, std::mt19937& rng, int n_players) {
    std::shuffle(deck.begin(), deck.end(), rng);
    std::vector<float> dealt_cards(deck.begin(), deck.begin() + n_players*2+5);
    return dealt_cards;
}

void weight_update(std::vector<Matrix>& WN, std::vector<float> input, int hidden_layer_size, int n_hidden_layers, std::vector<float> output, float n, double t, double t_mean, float& a0) {
    // forward pass
    std::vector<float> l1_w1 = input * WN[0];
    std::vector<float> a1(hidden_layer_size + 1);
    std::transform(l1_w1.begin(), l1_w1.end(), a1.begin(), f_max);
    a1[hidden_layer_size] = 1;
    std::vector<std::vector<float>> an;
    an.push_back(a1);
    for (int k = 0; k < n_hidden_layers - 1; k++) {
        std::vector<float> lk_wk = an[k] * WN[k + 1];
        std::vector<float> ak(hidden_layer_size + 1);
        std::transform(lk_wk.begin(), lk_wk.end(), ak.begin(), f_max);
        ak[hidden_layer_size] = 1;
        an.push_back(ak);
    }
    std::vector<float> lfinal_wfinal = an.back() * WN[n_hidden_layers];
    std::vector<float> afinal = lfinal_wfinal;


    //loss per image calc
    std::vector<float> loss;
    for (int k = 0; k < 5; k++) {
        a0 += (output[k] - afinal[k]) * (output[k] - afinal[k]);
        loss.push_back((afinal[k] - output[k]) * 2*(float)(t+1)/(float)t_mean);
    }

    //

    //


    // backward pass (ussing cross entropy+sigmoid)
    std::vector<float> Deltafinal = loss;
    std::vector < std::vector<float>> Delta_k;
    Delta_k.push_back(Deltafinal);
    an.push_back(afinal);
    //Delta2 Calcs
    for (int k = 0; k < n_hidden_layers; k++) {

        const Matrix& Wk = WN[n_hidden_layers - k];
        std::vector<float> ak = an[an.size() - 2 - k];
        std::vector<float> Sig_Delta(ak.size());
        std::transform(ak.begin(), ak.end(), Sig_Delta.begin(), delta_f_max);
        Sig_Delta[hidden_layer_size] = 0.0f;
        std::vector<float> Deltak = matrix_vec_multiply_transposed(Delta_k[k], Wk , Sig_Delta);
        Delta_k.push_back(Deltak);
    }
    //grad W1 W2
    std::vector<std::vector<float>> feeding_activations;
    for (int k = n_hidden_layers - 1; k >= 0; k--) feeding_activations.push_back(an[k]);
    feeding_activations.push_back(input);

    int S = Delta_k.size();
    for (int k = 0; k < S; k++) {
        WN[k].subtracing_scaled_matrix_times_vec(feeding_activations[S - 1 - k], Delta_k[S - 1 - k], n);
    }
};




void write_int(std::ofstream& out, int v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(int));
}

static int read_int(std::ifstream& out) {
    int v = 0;
    out.read(reinterpret_cast<char*>(&v), sizeof(int));
    return v;
}


void write_WN(std::ofstream& out, const std::vector<Matrix>& WN) {
    write_int(out, WN.size());
    for (int k = 0; k < WN.size(); k++) {
        const Matrix& WN_k = WN[k];
        int rows = WN_k.rows();
        int cols = WN_k.cols();
        write_int(out, rows);
        write_int(out, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float v = WN_k.Extract_Value(i, j);
                out.write(reinterpret_cast<const char*>(&v), sizeof(float));
            }
        }
    }
}

std::vector<Matrix> read_WN(std::ifstream& in) {
    int K = read_int(in);
    std::vector<Matrix> WN;
    for (int k = 0; k < K; k++) {
        std::size_t rows = read_int(in);
        std::size_t cols = read_int(in);
        Matrix WN_k(rows, cols, 0.0f);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float v = 0;
                in.read(reinterpret_cast<char*>(&v), sizeof(float));
                WN_k.Extract_Value(i, j) = v;
            }
        }
        WN.push_back(std::move(WN_k));
    }
    return WN;
}


void WN_save(const std::vector<Matrix>& WN, const std::string& filname) {
    std::ofstream out(filname, std::ios::binary);
    if (!out) throw std::runtime_error("could not open");
    write_WN(out, WN);
}


std::vector<Matrix> WN_read(const std::string& filname) {
    std::ifstream in(filname, std::ios::binary);
    if (!in) throw std::runtime_error("could not open");
    return read_WN(in);
}

std::vector<Matrix> WN_initilizer(int N, int hidden_layer_size, int n_hidden_layers, int final_layer_size) {
    std::vector<Matrix> WN;



    //initalise network;
    Matrix Wfirst = Matrix::Random_Intaliser(N + 1, hidden_layer_size + 1);//+1 for biase

    WN.push_back(Wfirst);
    for (int i = 0; i < n_hidden_layers - 1; i++) {
        WN.push_back(Matrix::Random_Intaliser(hidden_layer_size + 1, hidden_layer_size + 1));
    }
    Matrix Wlast = Matrix::Random_Intaliser(hidden_layer_size + 1, final_layer_size);//+1 for biase
    WN.push_back(Wlast);
    return WN;
}



std::vector<float> forward_pass(std::vector<Matrix>& WN, std::vector<float> input, int hidden_layer_size, int n_hidden_layers) {
    // forward pass
    std::vector<float> l1_w1 = input * WN[0];
    std::vector<float> a1(hidden_layer_size + 1);
    std::transform(l1_w1.begin(), l1_w1.end(), a1.begin(), f_max);
    a1[hidden_layer_size] = 1;
    std::vector<std::vector<float>> an;
    an.push_back(a1);
    for (int k = 0; k < n_hidden_layers - 1; k++) {
        std::vector<float> lk_wk = an[k] * WN[k + 1];
        std::vector<float> ak(hidden_layer_size + 1);
        std::transform(lk_wk.begin(), lk_wk.end(), ak.begin(), f_max);
        ak[hidden_layer_size] = 1;
        an.push_back(ak);
    }
    std::vector<float> lfinal_wfinal = an.back() * WN[n_hidden_layers];
    return lfinal_wfinal;
}


int main() {
    int threads = 4;
    int max_capacity = 200000;
    int strategy_capacity = 200000;
    int n_walks = 120;
    int n_generations = 100;
    int seed_n = 7;
    int gernerations = 15;
    int Cards_Dealt_per_loop = 2000;
    int strategy_generations = 10;

    omp_set_num_threads(threads);
    std::cout << omp_get_max_threads() << " threads\n";
    std::mt19937 shuffle_rng(seed_n);

    int max_rounds = 2;
    int n_players = 2;
    int C = 3;
    int max_actions_r1 = (C+1)*(n_players-1)+1;
    int max_actions_r2_4 = (C + 2) * (n_players - 1) + 1;

    int b0 = 2;
    float max_player_contribution = 0;
    for (int r = 1; r <= max_rounds; r++) {
        float unit = (r <= 2) ? b0 : 2 * b0;
        max_player_contribution += (C + 1) * unit;
    }
    float max_pot = n_players * max_player_contribution;
    int n_actions = 5;
    int ranks = 4;
    std::vector<float> gamstate_vec(42 + 7*n_players + (max_actions_r1 + (max_rounds - 1) * max_actions_r2_4)*(n_actions+n_players), 0.0f);
    gamstate_vec[0] = 1;//round 1
    int pot = 3;
    gamstate_vec[1] = pot;
    
    for (int i = 0; i < n_players; i++) {
        gamstate_vec[40 + 4 * n_players + i*3 + 2] = 1;//all player playing
    }

    int button = n_players-1;
    int sb, bb, first_preflop;
    if (n_players == 2) { sb = button; bb = (button + 1) % n_players; first_preflop = sb; }
    else { sb = (button + 1) % n_players; bb = (button + 2) % n_players; first_preflop = (button + 3) % n_players; }

    gamstate_vec[40 + 2 * n_players + first_preflop] = 1;//player preflop turn
    gamstate_vec[40 + n_players + button] = 1;
    //player contributions (blinds)
    gamstate_vec[40 + 3 * n_players + bb] = 2;
    gamstate_vec[40 + 3 * n_players + sb] = 1;

    gamstate_vec[40 + 7 * n_players] = 0;
    gamstate_vec[41 + 7 * n_players ] = n_players;

    //round (weight)
    //pot(weight) % of max pot
    //player-card (weight for rank + 1hot for suit)*2
    // ace_precent (binary)
    //public-card (weight for rank + 1hot for suit)*5
    // ace present (binary)
    //strength of best 5 (weigth) backcalcuated from hand number (higher weight = better hand) 
    //player-traversing (one hot)
    //player button (weight)*n_players
    //player-turn (one hot)
    //player-contribution (weight) % of max contribution*n_players
    //player-state (one-hot) *n_players
    //raises_made_this_round (weight)
    //n_players(weight)
    //round1-history (player-turn (one hot) ,player-action (one hot))*max round length
    //round2-history (player-turn (one hot) ,player-action (one hot))*max round length
    //round3-history (player-turn (one hot) ,player-action (one hot))*max round length
    //round4-history (player-turn (one hot) ,player-action (one hot))*max round length

    int n_cards = ranks*4;
    std::vector<float> player_contributions(n_players, 0);
    player_contributions[bb] = 2;
    player_contributions[sb] = 1;
    std::vector<std::string> player_status(n_players, "playing");
    std::vector<float> deck(n_cards);
    std::iota(deck.begin(), deck.end(), 0);




    std::vector<std::vector<float>> regret_gamstate_training;
    std::vector<std::vector<float>> stratagy_gamstate_training;
    std::vector<std::vector<float>> regret_training;
    std::vector<std::vector<float>> stratagy_training;
    std::vector<int> regret_N_vec;
    std::vector<int> stratagy_N_vec;

    




    std::mt19937 rng(42);
    std::mt19937 reservoir_rng(123);

    int last_agressor = bb;

    std::vector<std::string> history = {};
    std::vector<std::string> round_history = {};

    int player_turn = first_preflop;
    int round = 1;
    int  round_actions_count = 0;
    

    int N = gamstate_vec.size();
    
    int alpha = 2;
    int W_min = 128;
    int W_max = 1024;
    int hidden_layer_size = std::clamp(alpha * N, W_min, W_max);
    int L_min = 2;
    int n_hidden_layers = L_min + floor(log2(n_players));
    int final_layer_size = 5;

    int n_per_network_update = 1;

    int P1 = ranks;
    int P2 = P1 * ranks;
    int P3 = P2 * ranks;
    int P4 = P3 * ranks;
    int P5 = P4 * ranks;

    std::vector <uint32_t> straight_lookup = build_straight_lookup(ranks);

    float n0 = 0.01;
    float f = 0.1;
    float q = (1 / f - 1) / gernerations;
    int network_count = 0;
    

    std::string file_name = "WN";

    std::vector<Matrix> WN = WN_initilizer(N, hidden_layer_size, n_hidden_layers, final_layer_size);
    

    


    


    std::unordered_map<int, float>hand_lookup_table = build_hand_lookup_table(ranks, straight_lookup, P5, P4, P3, P2, P1);


    //######################################################
    //############################################################

    

    long long N_regret_samples_colected = 0;
    long long N_stratagy_samples_colected = 0;
    
    double traverse_sec = 0, train_sec = 0;
    std::cout << "threads = " << threads <<"  seed = "<< seed_n <<"  capacity = " << max_capacity << std::endl;


    
    for (int j = 0; j < n_generations; j++) {
        auto t_a = std::chrono::steady_clock::now();
        for (int i = 0; i < Cards_Dealt_per_loop; i++) {
            std::vector<float> cards_ = hand(deck, rng,n_players);


            int player_traversing = (i + j) % n_players;
            std::vector<float> new_gamstate_vec = gamstate_vec;
            Decition_Node node_root = Decition_Node(last_agressor, cards_, n_players, history, round_history, round, player_contributions, pot, player_status, player_turn, new_gamstate_vec, round_actions_count, max_actions_r1, player_traversing, ranks, straight_lookup, P5,P4,P3,P2,P1, max_actions_r2_4,0, C, b0, max_rounds);
            node_root.nerual_network_recursion( player_traversing, rng, regret_gamstate_training, regret_training, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_regret_samples_colected, C, hand_lookup_table, N_stratagy_samples_colected, stratagy_gamstate_training, regret_N_vec, stratagy_N_vec, j, strategy_capacity, reservoir_rng);
        }
        traverse_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_a).count();

        if (j % n_per_network_update == 0) {

            WN = WN_initilizer(N, hidden_layer_size, n_hidden_layers, final_layer_size);
            auto t_b = std::chrono::steady_clock::now();
            
            network_count = 0;
            double t_mean_reg = 0;
            
            std::vector<int> regret_all_order(regret_gamstate_training.size());
            std::iota(regret_all_order.begin(), regret_all_order.end(), 0);
            std::shuffle(regret_all_order.begin(), regret_all_order.end(), shuffle_rng);

            std::size_t n_val = regret_all_order.size() / 10;

            std::vector<int> regret_val_order(regret_all_order.begin(), regret_all_order.begin() + n_val);
            std::vector<int> regret_train_order(regret_all_order.begin() + n_val, regret_all_order.end());

            std::vector<float> train_mean(5, 0.0f);

            std::vector<float> val_mean(5, 0.0f);
            for (int i = 0; i < regret_val_order.size(); i++) {
                for (int j = 0; j < 5; j++) {
                    val_mean[j] += regret_training[regret_val_order[i]][j]/max_pot;
                }
            }
            for (int j = 0; j < 5; j++) val_mean[j] /= regret_val_order.size();

            float val_target_var = 0;
            for (int i = 0; i < regret_val_order.size(); i++) {
                for (int j = 0; j < 5; j++) {
                    float d = regret_training[regret_val_order[i]][j] / max_pot - val_mean[j];
                    val_target_var += d * d;
                }
            }
            val_target_var /= (regret_val_order.size() * 5);



            for (int i = 0; i < regret_train_order.size(); i++) {
                for (int a = 0; a < 5; a++) {
                    train_mean[a] += regret_training[regret_train_order[i]][a] / max_pot;
                }
            }
            for (int a = 0; a < 5; a++) train_mean[a] /= regret_train_order.size();

            float train_target_var = 0;
            for (int i = 0; i < regret_train_order.size(); i++) {
                for (int a = 0; a < 5; a++) {
                    float d = regret_training[regret_train_order[i]][a] / max_pot - train_mean[a];
                    train_target_var += d * d;
                }
            }
            train_target_var /= (regret_train_order.size() * 5);


            for (int i = 0; i < regret_gamstate_training.size(); i++) {
                t_mean_reg += (regret_N_vec[i] + 1);
            }
            t_mean_reg /= regret_gamstate_training.size();

            

            if (!std::isfinite(t_mean_reg)) {
                throw std::runtime_error("t_mean non-positive: reg=" + std::to_string(t_mean_reg));
            }

            std::cout << "regret_gamstate_training.size() and stratagy_gamstate_training.size() = " << regret_gamstate_training.size() << " " << stratagy_gamstate_training.size() << std::endl;


            float best_val = 1e30f;
            int best_epoch = -1;
            std::vector<Matrix> best_WN = WN;

            for (int k = 0; k < gernerations; k++) {
                network_count++;

                std::shuffle(regret_train_order.begin(), regret_train_order.end(), shuffle_rng);
                float a0_regret = 0;
                
                float n = n0 / (1 + q * network_count);

                
                

                
                int n_train = (int)regret_train_order.size();
                int n_serial = (k == 0) ? std::min(1000, n_train) : 0;

                for (int i = 0; i < n_serial; i++) { 
                    int idx = regret_train_order[i];
                    std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, regret_gamstate_training[idx], C, hand_lookup_table, max_rounds);
                    input.push_back(1);
                    std::vector<float> output = regret_normalisation(max_pot, regret_training[idx]);
                    weight_update(WN, input, hidden_layer_size, n_hidden_layers, output, n, regret_N_vec[idx], t_mean_reg, a0_regret);
                }
#pragma omp parallel for schedule(static) reduction(+:a0_regret)
                for (int i = n_serial; i < n_train; i++) {
                    int idx = regret_train_order[i];
                    std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, regret_gamstate_training[idx], C, hand_lookup_table, max_rounds);
                    input.push_back(1);
                    std::vector<float> output = regret_normalisation(max_pot, regret_training[idx]);
                    weight_update(WN, input, hidden_layer_size, n_hidden_layers, output, n, regret_N_vec[idx], t_mean_reg, a0_regret);


                }
                
                

                float reg_err = 0;
                int n_valid = (int)regret_val_order.size();
#pragma omp parallel for schedule(static) reduction(+:reg_err)
                for (int i = 0; i < n_valid; i++) {
                    int idx = regret_val_order[i];
                    std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, regret_gamstate_training[idx], C, hand_lookup_table, max_rounds);
                    input.push_back(1);
                    std::vector<float> prediction = forward_pass(WN, input, hidden_layer_size, n_hidden_layers);
                    std::vector<float> target = regret_normalisation(max_pot, regret_training[idx]);
                    for (int j = 0; j < 5; j++) {
                        reg_err += (target[j] - prediction[j]) * (target[j] - prediction[j]);
                    }
                }
                float val_ratio = reg_err / (regret_val_order.size() * 5) / val_target_var;

                a0_regret /= (regret_train_order.size() * 5);
                std::cout <<"j = " << j << "  k = " << k << "  train = " << a0_regret / train_target_var << "  val = " << val_ratio << std::endl;
                if (val_ratio < best_val) {
                    best_val = val_ratio; best_epoch = k; best_WN = WN  ;
                    
                }




                
            }


            


            if (best_epoch == -1) throw std::runtime_error("value net diverged at j = " + std::to_string(j));
            train_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_b).count();
            WN = best_WN;
            int z = WN.size();
            for (int i = 0; i < z; i++) {
                const Matrix& WN_i = WN[i];
                int x = WN_i.rows();
                int y = WN_i.cols();
                for (int j = 0; j < x; j++) {
                    for (int k = 0; k < y; k++) {
                        if (!std::isfinite(WN_i.Extract_Value(j, k))) {
                            throw std::runtime_error("zero erro matrix WN");
                        }
                    }
                }
            }
            WN_save(WN, file_name + "_" + std::to_string(j) + " max_rounds " + std::to_string(max_rounds) + ".bin");
            std::cout << "best epoch = " << best_epoch << "  best val = " << best_val << std::endl;
        }
        


    }

    //#####################################################################################################
    //#####################################################################################################
    //#####################################################################################################
    std::cout << "traverse_sec, train_sec = " << traverse_sec << " " << train_sec << std::endl;

    std::vector<Matrix> WN_stratagy = WN_initilizer(N, hidden_layer_size, n_hidden_layers, final_layer_size);
    double t_mean_stratagy = 0;
    
    for (int i = 0; i < stratagy_gamstate_training.size(); i++) {
        t_mean_stratagy += (stratagy_N_vec[i] + 1);
    }
    t_mean_stratagy /= stratagy_gamstate_training.size();

    




    network_count = 0;



    std::vector<int> all_order(stratagy_gamstate_training.size());
    std::iota(all_order.begin(), all_order.end(), 0);          
    std::shuffle(all_order.begin(), all_order.end(), shuffle_rng);

    std::size_t n_val = all_order.size() / 10;              

    std::vector<int> val_order(all_order.begin(), all_order.begin() + n_val);
    std::vector<int> train_order(all_order.begin() + n_val, all_order.end());



    std::vector<float> val_mean(5, 0.0f);
    for (int i = 0; i < val_order.size(); i++) {
        for (int j = 0; j < 5; j++) {
            val_mean[j] += stratagy_training[val_order[i]][j];
        }
    }
    for (int j = 0; j < 5; j++) val_mean[j] /= val_order.size();

    float val_target_var = 0;
    for (int i = 0; i < val_order.size(); i++) {
        for (int j = 0; j < 5; j++) {
            float d = stratagy_training[val_order[i]][j] - val_mean[j];
            val_target_var += d * d;
        }
    }
    val_target_var /= (val_order.size() * 5);


    std::vector<float> train_mean_s(5, 0.0f);
    for (int i = 0; i < train_order.size(); i++) {
        for (int a = 0; a < 5; a++) {
            train_mean_s[a] += stratagy_training[train_order[i]][a];
        }
    }
    for (int a = 0; a < 5; a++) train_mean_s[a] /= train_order.size();

    float train_target_var_s = 0;
    for (int i = 0; i < train_order.size(); i++) {
        for (int a = 0; a < 5; a++) {
            float d = stratagy_training[train_order[i]][a] - train_mean_s[a];
            train_target_var_s += d * d;
        }
    }
    train_target_var_s /= (train_order.size() * 5);

    float q_strat = (1 / f - 1) / strategy_generations;

    float best_val = 1e9f;
    int best_epoch = -1;
    std::vector<Matrix> best_WN = WN_stratagy;

    for (int k = 0; k < strategy_generations; k++) {

        network_count++;
        std::shuffle(train_order.begin(), train_order.end(), shuffle_rng);
        float n = n0 / (1 + q_strat * network_count);
        float a0_strat = 0;

        int n_train_s = (int)train_order.size();
        int n_serial_s = (k == 0) ? std::min(1000, n_train_s) : 0;

        for (int i = 0; i < n_serial_s; i++) {
            int idx = train_order[i];
            std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, stratagy_gamstate_training[idx], C, hand_lookup_table, max_rounds);
            input.push_back(1);
            std::vector<float> output_stratagy = stratagy_training[idx];
            weight_update(WN_stratagy, input, hidden_layer_size, n_hidden_layers, output_stratagy, n, stratagy_N_vec[idx], t_mean_stratagy, a0_strat);
        }
#pragma omp parallel for schedule(static) reduction(+:a0_strat)
        for (int i = n_serial_s; i < n_train_s; i++) {
            int idx = train_order[i];
            std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, stratagy_gamstate_training[idx], C, hand_lookup_table, max_rounds);
            input.push_back(1);
            std::vector<float> output_stratagy = stratagy_training[idx];
            weight_update(WN_stratagy, input, hidden_layer_size, n_hidden_layers, output_stratagy, n, stratagy_N_vec[idx], t_mean_stratagy, a0_strat);
        }

        float val_err = 0;
        int n_valid_s = (int)val_order.size();
#pragma omp parallel for schedule(static) reduction(+:val_err)
        for (int i = 0; i < n_valid_s; i++) {
            int idx = val_order[i];
            std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, stratagy_gamstate_training[idx], C, hand_lookup_table, max_rounds);
            input.push_back(1);
            std::vector<float> prediction = forward_pass(WN_stratagy, input, hidden_layer_size, n_hidden_layers);
            const std::vector<float>& target = stratagy_training[idx];
            for (int j = 0; j < 5; j++) {
                val_err += (target[j] - prediction[j]) * (target[j] - prediction[j]);
            }
        }
        float val_ratio = val_err / (val_order.size() * 5) / val_target_var;

        a0_strat /= (train_order.size() * 5);
        std::cout << "k = " << k << "  train = " << a0_strat / train_target_var_s << "  val = " << val_ratio << std::endl;
        if (val_ratio < best_val) { 
            best_val = val_ratio; best_epoch = k; best_WN = WN_stratagy; 
        }



        

        int z = WN_stratagy.size();
        for (int i = 0; i < z; i++) {
            const Matrix& WN_i = WN_stratagy[i];
            int x = WN_i.rows();
            int y = WN_i.cols();
            for (int j = 0; j < x; j++) {
                for (int k = 0; k < y; k++) {
                    if (!std::isfinite(WN_i.Extract_Value(j, k))) {
                        throw std::runtime_error("zero erro matrix WN_stratagy");
                    }
                }
            }
        }


    }
    WN_stratagy = best_WN;
    WN_save(WN_stratagy, "WN_stratagy record");
    std::cout << "best epoch = " << best_epoch << "  best val = " << best_val << std::endl;



    //exploitablity (average across all players)
    std::vector <std::vector<int>> predetor_possible_hands;
    for (int i = 0; i < ranks * 4 ; i++) {
        for (int j = i+1 ; j < ranks * 4 ; j++) {
            predetor_possible_hands.push_back({ i,j });
        }
    }

    std::mt19937 eval_rng(12345);


    std::vector<std::vector<int>> sampled;
    std::sample(predetor_possible_hands.begin(), predetor_possible_hands.end(),
    std::back_inserter(sampled), n_walks, eval_rng);

    predetor_possible_hands = sampled;


    std::unordered_map<std::string, std::vector<float>> strat_cache;
    auto t0 = std::chrono::steady_clock::now();
    int S = predetor_possible_hands.size();
    float average_exploitablity = 0;
    float exploitablity;
    std::vector<float> exploitablity_vec;
    for (int i = 0; i < n_players; i++) {
        exploitablity = 0;
        for (int j = 0; j < S; j++) {
            std::vector<float> cards_(n_players * 2 + 5, 0);
            int card1 = predetor_possible_hands[j][0];
            int card2 = predetor_possible_hands[j][1];
            cards_[i * 2] = card1;
            cards_[i * 2 + 1] = card2;

            std::vector <std::vector<float>> possible_hands;
            for (int l = 0; l < ranks * 4 ; l++) {
                for (int m = l + 1; m < ranks * 4 ; m++) {
                    if (l != card1 && l != card2 && m != card1 && m != card2) {
                        possible_hands.push_back({ (float)l,(float)m });
                    }
                }
            }

            std::vector <std::vector<float>> flops;
            for (int l = 0; l < ranks * 4 ; l++) {
                for (int m = l +1; m < ranks * 4 ; m++) {
                    for (int n = m + 1; n < ranks * 4; n++) {
                        if (l != card1 && l != card2 && m != card1 && m != card2 && n != card1 && n != card2) {
                            flops.push_back({ (float)l,(float)m,(float)n });
                        }
                    }
                }
            }


            std::vector<float> weight_vec(possible_hands.size(), 1.0f / possible_hands.size());

            int player_traversing = i;
            std::vector<float> new_gamstate_vec = gamstate_vec;
            Decition_Node node_root = Decition_Node(last_agressor, cards_, n_players, history, round_history, round, player_contributions, pot, player_status, player_turn, new_gamstate_vec, round_actions_count, max_actions_r1, player_traversing, ranks, straight_lookup, P5, P4, P3, P2, P1, max_actions_r2_4, 0, C, b0, max_rounds);

            exploitablity_vec = node_root.exploitiblity_calc(i, predetor_possible_hands[j], possible_hands, weight_vec, ranks, flops, WN_stratagy,hidden_layer_size,  n_hidden_layers,  max_pot,  max_player_contribution,  C, hand_lookup_table, strat_cache);
            std::cout << "exploitablity "<<j <<" = " << exploitablity_vec[i] << std::endl;
            exploitablity += exploitablity_vec[i];

        }
        average_exploitablity += (exploitablity / (n_players*S));
    }
    std::cout << "average_exploitablity = " << average_exploitablity << std::endl;
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << n_walks*2 << " walks took " << secs << std::endl;



    //is stratagy resonable (give specific hand)
    //#########################################################################################################################################
    //#########################################################################################################################################
    //#########################################################################################################################################

    std::vector <float> cards(7, 0);
    int player_traversing = first_preflop;
    std::vector<std::vector<int>> trial_hands = {{12,13},{0,5},{4,9}};
    for (int i = 0; i < trial_hands.size(); i++) {
        cards[first_preflop * 2] = trial_hands[i][0];
        cards[first_preflop * 2 + 1] = trial_hands[i][1];
        std::vector<float> new_gamstate_vec = gamstate_vec;
        Decition_Node node_root = Decition_Node(last_agressor, cards, n_players, history, round_history, round, player_contributions, pot, player_status, player_turn, new_gamstate_vec, round_actions_count, max_actions_r1, player_traversing, ranks, straight_lookup, P5, P4, P3, P2, P1, max_actions_r2_4, 0, C, b0, max_rounds);

        std::vector<float> probs = node_root.handresponse(ranks, WN_stratagy, hidden_layer_size, n_hidden_layers, max_pot, max_player_contribution, C, hand_lookup_table);

        std::vector<std::string> action_names = { "fold", "check", "call", "bet", "raise" };
        std::cout << "cards " << trial_hands[i][0] << "," << trial_hands[i][1] << " : ";
        for (int a = 0; a < 5; a++) {
            std::cout << action_names[a] << "=" << probs[a] << "  ";
        }
        std::cout << std::endl;
    }



    return 0;
} 