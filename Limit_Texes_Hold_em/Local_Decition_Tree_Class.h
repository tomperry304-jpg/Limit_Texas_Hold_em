#pragma once
#include "Matrix_Class.h"
#include <vector>
#include <cstddef>
#include <string>
#include <iostream>
#include <stdexcept>
#include <omp.h>  
#include <random>
#include <unordered_map>
#include <numeric>


float f_max(float z);
std::vector<float> gamstate_normalisation(float max_pot, float max_player_contribution, int ranks, int n_players, std::vector<float> gamestate_vec, int C, const std::unordered_map<int, float>& hand_lookup_table, int max_rounds);


class Decition_Node {
private:
    std::vector<float> cards_;
    std::vector<std::string> history_;
    std::vector<std::string> round_history_;
    int round_;
    std::vector<float> player_contributions_;
    int n_players_;
    float pot_;
    std::vector<std::string> player_status_;
    int player_turn_;
    int last_agressor_;
    std::vector<float> gamestate_vec_;
    int round_actions_count_;
    int max_actions_r1_;
    int player_traversing_;
    static inline const std::unordered_map<std::string, int> action_index_ = {
    {"fold", 0}, {"check", 1}, {"call", 2}, {"bet", 3}, {"raise", 4}
    };
    int ranks_;
    std::vector <uint32_t> strait_lookup_;
    long long P5_;
    long long P4_;
    long long P3_;
    long long P2_;
    long long P1_;
    int max_actions_r2_4_;
    int raises_this_round_;
    int C_;
    float b0_;
    int max_rounds_;
public:
    Decition_Node(int last_agressor, const std::vector<float>& cards, int n_players, const std::vector<std::string>& history, const std::vector<std::string>& round_history, int round, const std::vector<float>& player_contributions, float pot, const std::vector<std::string>& player_status, int player_turn, const std::vector<float>& gamestate_vec, int round_actions_count, int max_actions_r1, int player_traversing, int ranks, const std::vector <uint32_t>& strait_lookup, const long long P5, const long long P4, const long long P3, const long long P2, const long long P1, int max_actions_r2_4, int raises_this_round, int C, int b0, int max_rounds)
        : cards_(cards), history_(history), round_history_(round_history), round_(round),  player_contributions_(player_contributions), n_players_(n_players), pot_(pot), player_status_(player_status), player_turn_(player_turn), last_agressor_(last_agressor), gamestate_vec_(gamestate_vec), round_actions_count_(round_actions_count), max_actions_r1_(max_actions_r1), player_traversing_(player_traversing), ranks_(ranks), strait_lookup_(strait_lookup), P5_(P5), P4_(P4), P3_(P3),P2_(P2),P1_(P1), max_actions_r2_4_(max_actions_r2_4), raises_this_round_(raises_this_round), C_(C), b0_(b0), max_rounds_(max_rounds){

        std::fill(gamestate_vec_.begin() + 2, gamestate_vec_.begin() + 12, 0.0f);
        gamestate_vec_[12] = 0.0f;
        for (int i = 0; i < 2; i++) {
            int card = cards[player_traversing_*2 + i];
            int suit = card % 4;
            int rank = card / 4;
            gamestate_vec_[2 + i * 5] = rank + 1;
            gamestate_vec_[2 + i * 5 + 1 + suit] = 1;
            if (rank == 12) {
                gamestate_vec_[12] = 1;
            }
        }
        std::fill(gamestate_vec_.begin() + 40, gamestate_vec_.begin() + 40 + n_players_, 0.0f);
        gamestate_vec_[40 + player_traversing_] = 1;
    }

    bool checking_check(int n_seats) {
        bool check = false;
        bool no_raise = true;
        int i = 2;
        int check_count = 1;
        while (no_raise) {
            if (i > (int)round_history_.size()) break;
            std::string action = round_history_[round_history_.size() - i];
            if (action == "check") {
                check_count += 1;
            }
            if (action == "raise") {
                no_raise = false;
                break;
            }
            if (check_count == n_seats) {
                check = true;
                break;
            }
            i += 1;
        }
        return check;
    }

    bool checking_call(int n_seats) {
        bool call = false;
        bool no_raise = true;
        int i = 1;
        int call_count = 0;
        while (no_raise) {
            if (i > (int)round_history_.size()) break;
            std::string action = round_history_[round_history_.size() - i];
            if (action == "call") {
                call_count += 1;
            }
            if (action == "raise") {
                no_raise = false;
                break;
            }
            if (call_count == n_seats) {
                call = true;
                break;
            }
            i += 1;
        }
        return call;
    }


    int n_seats_() {
        return std::count(player_status_.begin(), player_status_.end(), "playing");
    }

    std::vector<std::string> legal_actions() {
        if (round_history_.empty()) {
            if (round_ != 1) {
                return { "check","bet" };
            }
            else {
                return { "fold","call","raise" };
            }
        }
        int n_seats = n_seats_();
        if (n_seats == 1) {
            return {};
        }

        const std::string& last = round_history_.back();
        if (last == "check") {
            if (round_history_.size() >= n_seats && checking_check(n_seats)) return {};
            return { "check","bet" };
        }

        if (last == "bet") {
            return { "fold","call","raise" };
        }
        if (last == "raise") {
            if (raises_this_round_ >= C_) {
                return { "fold","call" };
            }
            else {
                return { "fold","call","raise" };
            }
        }
        if (last == "fold" || last == "call") {
            bool everyone_acted = (round_ != 1) || ((int)round_history_.size() >= n_seats);
            if ((player_turn_ == last_agressor_ && everyone_acted) || checking_call(n_seats)) return {};
            float highest_contribution = *std::max_element(player_contributions_.begin(), player_contributions_.end());
            if (player_contributions_[player_turn_] == highest_contribution) {
                if (raises_this_round_ >= C_) return {"call" };
                return {"call","raise" };
            }
            if (raises_this_round_ >= C_) return { "fold","call" };
            return { "fold","call","raise" };
        }
        return {};
    }

    int wining_card(int player) {
        if (round_ == 2) {
            std::vector<float> public_cards;
            for (int i = 0; i < 3; i++) {
                int card = cards_[n_players_ * 2 + i];
                public_cards.push_back(card);
            }
            public_cards.push_back(cards_[player * 2]);
            public_cards.push_back(cards_[player * 2 + 1]);
            return hand_evaluation(public_cards, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_);
        }
        else if (round_ == 3) {
            std::vector<float> public_cards;

            for (int i = 0; i < 4; i++) {
                int card = cards_[n_players_ * 2 + i];
                public_cards.push_back(card);
            }
            public_cards.push_back(cards_[player * 2]);
            public_cards.push_back(cards_[player * 2 + 1]);
            int highes_combo = -1;
            for (int i = 0; i < 6; i++) {
                std::vector<float> hand_;
                for (int j = 0; j < 6; j++) {
                    if (j != i) {
                        hand_.push_back(public_cards[j]);
                    }
                }
                int combo = hand_evaluation(hand_, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_);
                if (combo > highes_combo) {
                    highes_combo = combo;
                }
            }
            return highes_combo;
        }
        else {
            std::vector<float> public_cards;
            for (int i = 0; i < 5; i++) {
                int card = cards_[n_players_ * 2 + i];
                public_cards.push_back(card);
            }
            public_cards.push_back(cards_[player * 2]);
            public_cards.push_back(cards_[player * 2 + 1]);
            int highes_combo = -1;
            for (int i = 0; i < 7; i++) {
                for (int j = i + 1; j < 7; j++) {
                    std::vector<float> hand_;
                    for (int k = 0; k < 7; k++) {
                        if (k != i && k != j) {
                            hand_.push_back(public_cards[k]);
                        }
                    }
                    int combo = hand_evaluation(hand_, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_);
                    if (combo > highes_combo) {
                        highes_combo = combo;
                    }
                }
            }
            return highes_combo;
        }
    }


    std::vector<float> terminal_returns() {
        std::vector<float> returns(player_contributions_.size(), 0.0f);
        int n_seats = std::count(player_status_.begin(), player_status_.end(), "playing");
        if (n_seats == 1) {
            auto it = std::find(player_status_.begin(), player_status_.end(), "playing");
            int winner_idx = std::distance(player_status_.begin(), it);
            for (int i = 0; i < n_players_; i++) {
                if (i == winner_idx) {
                    returns[i] = pot_ - player_contributions_[i];
                }
                else {
                    returns[i] = -1 * player_contributions_[i];
                }
            }
        }
        else {
            int high_score = -1;
            int player_score;
            std::vector<int> winner_i;
            for (int i = 0; i < n_players_; i++) {
                returns[i] = -1 * player_contributions_[i];
                if (player_status_[i] == "playing") {
                    player_score = wining_card(i);
                    if (player_score > high_score) {
                        high_score = player_score;
                        winner_i.clear();
                        winner_i.push_back(i);
                    }
                    else if (player_score == high_score) {
                        winner_i.push_back(i);
                    }
                }
            }
            for (int i = 0; i < winner_i.size(); i++) {
                returns[winner_i[i]] += pot_ / winner_i.size();
            }
        }
        return returns;
    }


    Decition_Node apply_action(std::string action) {
        //cards_, history_, round_history_, round_, n_players_, player_contributions_, pot_, player_status_, player_turn_, last_agressor_;
        std::vector<float> new_gamestate_vec_ = gamestate_vec_;
        new_gamestate_vec_[40 +2*n_players_+ player_turn_] = 0;
        int action_i = 1;
        float new_pot_ = pot_;
        std::vector<float> new_player_contributions_ = player_contributions_;
        std::vector<std::string> new_player_status_ = player_status_;
        std::vector<std::string> new_history_ = history_;
        std::vector<std::string> new_round_history_ = round_history_;
        int new_last_agressor_ = last_agressor_;
        int new_player_turn_ = player_turn_;
        int new_raises_this_round_ = raises_this_round_;
        if (action == "raise" || action == "call" || action == "bet") {
            //walk back untill hit
            bool hit = false;
            int i = 1;
            int round_multiple = b0_*2;
            if (round_ == 1 || round_ == 2)
                round_multiple = b0_;
            while (hit == false) {
                if (i > n_players_) throw std::runtime_error("no active player found");
                else if (new_player_status_[((player_turn_ - i) % n_players_ + n_players_) % n_players_] == "playing") {
                    hit = true;
                }
                
                else {
                    i++;
                }
            }
            int I = ((player_turn_ - i) % n_players_ + n_players_) % n_players_;
            new_pot_ -= new_player_contributions_[player_turn_];
            if (action == "raise") {
                new_player_contributions_[player_turn_] = new_player_contributions_[I] + round_multiple;
                new_last_agressor_ = player_turn_;
                action_i = 4;
                new_gamestate_vec_[40 + 7 * n_players_] += 1;
                new_raises_this_round_ += 1;
            }
            else if (action == "bet") {
                new_player_contributions_[player_turn_] = new_player_contributions_[I] + round_multiple;
                new_last_agressor_ = player_turn_;
                
                action_i = 3;
            }
            else {
                new_player_contributions_[player_turn_] = new_player_contributions_[I];
                action_i = 2;
            }
            new_pot_ += new_player_contributions_[player_turn_];
            new_gamestate_vec_[1] = new_pot_;
            new_gamestate_vec_[40 +3*n_players_ + player_turn_] = new_player_contributions_[player_turn_];
        }
        else if (action == "fold") {
            new_player_status_[player_turn_] = "folded";
            new_gamestate_vec_[40 + 4 * n_players_ + 3*player_turn_] = 1;
            new_gamestate_vec_[40 + 4 * n_players_ + 3 * player_turn_ + 2] = 0;
            new_gamestate_vec_[41 + 7 * n_players_] -= 1;
            action_i = 0;
        }
        new_history_.push_back(action);
        new_round_history_.push_back(action);
        if (round_ == 1) {
            new_gamestate_vec_[42 + 7 * n_players_ + round_actions_count_ * (n_players_ + 5) + player_turn_] = 1;
            new_gamestate_vec_[42 + 7 * n_players_ + round_actions_count_ * (n_players_ + 5) + n_players_ + action_i] = 1;
        }
        else {
            new_gamestate_vec_[42 + 7 * n_players_ + round_actions_count_ * (n_players_ + 5) + (max_actions_r1_ + (round_ - 2) * max_actions_r2_4_) * (n_players_ + 5) + player_turn_] = 1;
            new_gamestate_vec_[42 + 7 * n_players_ + round_actions_count_ * (n_players_ + 5) + (max_actions_r1_ + (round_ - 2) * max_actions_r2_4_) * (n_players_ + 5) + n_players_ + action_i] = 1;
        }
        //walk back untill hit
        bool hit = false;
        int i = 1;
        while (hit == false) {
            if (i > n_players_) throw std::runtime_error("no active player found");
            else if (new_player_status_[(player_turn_ + i) % n_players_] == "playing") {
                hit = true;
            }
            else {
                i++;
            }
        }
        new_player_turn_ = (player_turn_ + i) % n_players_;
        new_gamestate_vec_[40 + 2 * n_players_ + new_player_turn_] = 1;
        int new_round_actions_count = round_actions_count_ + 1;
        return Decition_Node(new_last_agressor_, cards_, n_players_, new_history_, new_round_history_, round_, new_player_contributions_, new_pot_, new_player_status_, new_player_turn_, new_gamestate_vec_, new_round_actions_count, max_actions_r1_, player_traversing_, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_, max_actions_r2_4_, new_raises_this_round_, C_, b0_, max_rounds_);
    }


    Decition_Node new_round() {
        std::vector<float> new_gamestate_vec_ = gamestate_vec_;
        //cards_, history_, round_history_, round_, n_players_, player_contributions_, pot_, player_status_, player_turn_, last_agressor_;
        std::vector<std::string> new_round_history_ = {};
        int new_player_turn_;
        int new_round_ = round_ + 1;

        new_gamestate_vec_[40 + 2 * n_players_ + player_turn_] = 0;
        bool hit = false;
        int j = 0;
        while (hit == false) {
            if (j >= n_players_) throw std::runtime_error("no active player found");
            else if (player_status_[j] == "playing") {
                hit = true;
            }
            
            else {
                j++;
            }
        }
        new_player_turn_ = j;
        new_gamestate_vec_[40 + 2 * n_players_ + j] = 1;
        new_gamestate_vec_[0] = new_round_;

        if (new_round_ == 2) {
            std::vector<float> public_cards;
            for (int i = 0; i < 3; i++) {
                int card = cards_[n_players_ * 2 + i];
                public_cards.push_back(card);
                int suit = card % 4;
                int rank = card / 4;
                new_gamestate_vec_[13 + i * 5] = rank + 1;
                new_gamestate_vec_[13 + i * 5 + 1 + suit] = 1;
                if (rank == 12) {
                    new_gamestate_vec_[38] = 1;
                }
            }
            public_cards.push_back(cards_[player_traversing_ * 2]);
            public_cards.push_back(cards_[player_traversing_ * 2 + 1]);
            new_gamestate_vec_[39] = hand_evaluation(public_cards, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_);
        }
        else if (new_round_ == 3) {
            int card = cards_[n_players_ * 2 + 3];
            int suit = card % 4;
            int rank = card / 4;
            new_gamestate_vec_[13 + 3 * 5] = rank + 1;
            new_gamestate_vec_[13 + 3 * 5 + 1 + suit] = 1;
            if (rank == 12) {
                new_gamestate_vec_[38] = 1;
            }
            std::vector<float> public_cards;
            
            for (int i = 0; i < 4; i++) {
                int card = cards_[n_players_ * 2 + i];
                public_cards.push_back(card);
            }
            public_cards.push_back(cards_[player_traversing_ * 2]);
            public_cards.push_back(cards_[player_traversing_ * 2 + 1]);
            int highes_combo = -1;
            for (int i = 0; i < 6; i++) {
                std::vector<float> hand_;
                for (int j = 0; j < 6; j++) {
                    if (j != i) {
                        hand_.push_back(public_cards[j]);
                    }
                }
                int combo = hand_evaluation(hand_, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_);
                if (combo > highes_combo) {
                    highes_combo = combo;
                }
            }
            new_gamestate_vec_[39] = highes_combo;
        }
        else {
            int card = cards_[n_players_ * 2 + 4];
            int suit = card % 4;
            int rank = card / 4;
            new_gamestate_vec_[13 + 4 * 5] = rank + 1;
            new_gamestate_vec_[13 + 4 * 5 + 1 + suit] = 1;
            if (rank == 12) {
                new_gamestate_vec_[38] = 1;
            }
            std::vector<float> public_cards;

            for (int i = 0; i < 5; i++) {
                int card = cards_[n_players_ * 2 + i];
                public_cards.push_back(card);
            }
            public_cards.push_back(cards_[player_traversing_ * 2]);
            public_cards.push_back(cards_[player_traversing_ * 2 + 1]);
            int highes_combo = -1;
            for (int i = 0; i < 7; i++) {
                for (int j = i+1; j < 7; j++) {
                    std::vector<float> hand_;
                    for (int k = 0; k < 7; k++) {
                        if (k != i && k != j) {
                            hand_.push_back(public_cards[k]);
                        }
                    }
                    int combo = hand_evaluation(hand_, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_);
                    if (combo > highes_combo) {
                        highes_combo = combo;
                    }
                }
            }
            new_gamestate_vec_[39] = highes_combo;
        }
        int new_round_actions_count = 0;
        new_gamestate_vec_[40 + 7 * n_players_] = 0;
        return Decition_Node(-1, cards_, n_players_, history_, new_round_history_, new_round_, player_contributions_, pot_, player_status_, new_player_turn_, new_gamestate_vec_, new_round_actions_count, max_actions_r1_, player_traversing_, ranks_, strait_lookup_, P5_, P4_, P3_, P2_, P1_, max_actions_r2_4_,0,C_,b0_, max_rounds_);
    }




    std::string get_key() {
        std::string key;
        if (cards_[2 * player_turn_] < cards_[2 * player_turn_ + 1]) {
            key += std::to_string((int)cards_[2 * player_turn_]) + "|Player" + std::to_string((int)cards_[2 * player_turn_ + 1]);
        }
        else {
            key += std::to_string((int)cards_[2 * player_turn_ + 1]) + "|Player" + std::to_string((int)cards_[2 * player_turn_]);
        }
        if (round_ >= 2) {
            std::vector<float> cards(cards_.begin() + n_players_ * 2, cards_.begin() + n_players_ * 2 + 3);
            std::sort(cards.begin(), cards.end());
            for (int i = 0; i < 3; i++) {
                key += "|Public" + std::to_string((int)cards[i]);
            }
            for (int i = 3; i < round_ + 1; i++) {
                key += "|Public" + std::to_string((int)cards_[2 * n_players_ + i]);
            }
            
        }
        for (std::string& a : history_) key += "|" + a;
        return key;
    }



    //####################################################################################################
    //####################################################################################################
    //####################################################################################################
    std::vector<float> player_centric_gamstate(int player_turn) {
        std::vector<float> player_gamestate_vec_ = gamestate_vec_;
        std::fill(player_gamestate_vec_.begin() + 2, player_gamestate_vec_.begin() + 12, 0.0f);
        player_gamestate_vec_[12] = 0;
        for (int i = 0; i < 2; i++) {
            int card = cards_[player_turn * 2 + i];
            int suit = card % 4;
            int rank = card / 4;
            player_gamestate_vec_[2 + i * 5] = rank + 1;
            player_gamestate_vec_[2 + i * 5 + 1 + suit] = 1;
            if (rank == 12) {
                player_gamestate_vec_[12] = 1;
            }
        }
        std::fill(player_gamestate_vec_.begin() + 40,player_gamestate_vec_.begin() + 40 + n_players_, 0.0f);
        player_gamestate_vec_[40 + player_turn] = 1;
        if (round_ > 1) {
            player_gamestate_vec_[39] = wining_card(player_turn);
        }
        else {
            player_gamestate_vec_[39] = 0;
        }
        return player_gamestate_vec_;
    }

    std::vector<float> forward_pass(std::vector<float> input, std::vector <Matrix>& WN, int hidden_layer_size, int n_hidden_layers, float max_pot) {
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
        for (int i = 0; i < 5; i++) {
            afinal[i] *= max_pot;
        }
        return afinal;
    }


    std::vector<float> nerual_network_recursion( int player_traverser, std::mt19937& rng, std::vector <std::vector<float>>& regret_gamstate_training, std::vector <std::vector<float>>& regret_training, float max_pot, float max_player_contribution, int ranks, std::vector<Matrix>& WN, int hidden_layer_size, int n_hidden_layers, std::vector <std::vector<float>>& stratagy_training, int max_capacity, long long& N_regret_samples_colected, int C, const std::unordered_map<int, float>& hand_lookup_table, long long& N_stratagy_samples_colected, std::vector <std::vector<float>>& stratagy_gamstate_training, std::vector<int>& regret_N_vec, std::vector<int>& stratagy_N_vec, int t, int strategy_capacity, std::mt19937& resivour_rng) {
        std::vector<std::string> actions = legal_actions();
        if (actions.empty()) {
            if (n_seats_() != 1 && round_ < max_rounds_) {
                Decition_Node child = new_round();
                return child.nerual_network_recursion( player_traverser, rng, regret_gamstate_training, regret_training, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_regret_samples_colected, C, hand_lookup_table, N_stratagy_samples_colected, stratagy_gamstate_training, regret_N_vec,stratagy_N_vec,t, strategy_capacity, resivour_rng);
            }
            else {
                return terminal_returns();
            }
        }
        else {
            std::vector<float> player_gamestate_vec;
            std::vector<float> node_value(n_players_, 0.0f);
            std::vector< std::vector<float>> exp_action_i;
            std::vector<float> input;
            if (player_turn_ == player_traverser) {
                input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, gamestate_vec_,C, hand_lookup_table, max_rounds_);
                input.push_back(1.0f);
            }
            else {
                player_gamestate_vec = player_centric_gamstate(player_turn_);
                input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, player_gamestate_vec, C, hand_lookup_table, max_rounds_);
                input.push_back(1.0f);
            }
            std::vector<float> raw = forward_pass(input, WN, hidden_layer_size, n_hidden_layers, max_pot);
            std::vector<float> regret(actions.size());
            for (int i = 0; i < actions.size(); i++) {
                float r = raw[action_index_.at(actions[i])];
                regret[i] = (r > 0.0f) ? r : 0.0f;
            }

            float total_regret = 0;
            for (int j = 0; j < regret.size(); j++) {
                if (regret[j] <= 0) {
                    regret[j] = 0;
                }
                total_regret += regret[j];
            }
            if (player_turn_ == player_traverser) {

                std::vector<float> regrets_training_node(5, 0);
                std::vector<float> gamstate_training_node = gamestate_vec_;
                for (int k = 0; k < actions.size(); k++) {
                    Decition_Node child = apply_action(actions[k]);

                    float p = 0;
                    if (total_regret == 0) {
                        int best = 0;
                        for (int i = 1; i < actions.size(); i++)
                            if (raw[action_index_.at(actions[i])] > raw[action_index_.at(actions[best])]) best = i;
                        p = (k == best) ? 1.0f : 0.0f;
                    }
                    else {
                        p = regret[k] / total_regret;
                    }



                    std::vector<float> exp = child.nerual_network_recursion( player_traverser, rng, regret_gamstate_training, regret_training, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_regret_samples_colected, C, hand_lookup_table, N_stratagy_samples_colected, stratagy_gamstate_training , regret_N_vec, stratagy_N_vec, t, strategy_capacity, resivour_rng);
                    exp_action_i.push_back(exp);
                    for (int l = 0; l < node_value.size(); l++) {
                        node_value[l] += p * exp[l];
                    }

                }
                for (int z = 0; z < actions.size(); z++) {
                    int action_i = action_index_.at(actions[z]);
                    regrets_training_node[action_i] = exp_action_i[z][player_turn_] - node_value[player_turn_];
                }
                N_regret_samples_colected += 1;

                if (N_regret_samples_colected <= max_capacity) {
                    regret_N_vec.push_back(t);
                    regret_gamstate_training.push_back(gamstate_training_node);
                    regret_training.push_back(regrets_training_node);
                }
                else {
                    std::vector<float> p_dist0;
                    float p = (float)max_capacity / (float)N_regret_samples_colected;
                    p_dist0 = { p, 1 - p };
                    std::discrete_distribution<int> dist(p_dist0.begin(), p_dist0.end());
                    int chosen0 = dist(resivour_rng);
                    if (chosen0 == 0) {
                        std::uniform_int_distribution<int> dist(0, max_capacity - 1);
                        int I = dist(resivour_rng);
                        regret_N_vec[I] = t;
                        regret_gamstate_training[I] = gamstate_training_node;
                        regret_training[I] = regrets_training_node;
                    }
                }
            }
            else {
                


                std::vector<float> p_dist;
                p_dist.resize(actions.size());
                if (total_regret == 0) {
                    int best = 0;
                    for (int i = 1; i < actions.size(); i++)
                        if (raw[action_index_.at(actions[i])] > raw[action_index_.at(actions[best])]) best = i;
                    for (int x = 0; x < actions.size(); x++) {
                        p_dist[x] = (x == best) ? 1.0f : 0.0f;
                    }
                }
                else {
                    for (int x = 0; x < actions.size(); x++) {
                        p_dist[x] = regret[x] / total_regret;
                    }
                }

                std::vector<float> stratagy_training_node(5, 0);
                for (int i = 0; i < actions.size(); i++) {
                    stratagy_training_node[action_index_.at(actions[i])] = p_dist[i];
                }

                N_stratagy_samples_colected += 1;
                if (N_stratagy_samples_colected <= strategy_capacity) {
                    stratagy_N_vec.push_back(t);
                    stratagy_gamstate_training.push_back(player_gamestate_vec);
                    stratagy_training.push_back(stratagy_training_node);
                }
                else {
                    std::vector<float> p_dist0;
                    float p = (float)strategy_capacity / (float)N_stratagy_samples_colected;
                    p_dist0 = { p, 1 - p };
                    std::discrete_distribution<int> dist(p_dist0.begin(), p_dist0.end());
                    int chosen0 = dist(resivour_rng);
                    if (chosen0 == 0) {
                        std::uniform_int_distribution<int> dist(0, strategy_capacity - 1);
                        int I = dist(resivour_rng);
                        stratagy_N_vec[I] = t;
                        stratagy_gamstate_training[I] = player_gamestate_vec;
                        stratagy_training[I] = stratagy_training_node;
                    }
                }
                std::discrete_distribution<int> dist(p_dist.begin(), p_dist.end());
                int chosen = dist(rng);
                Decition_Node child = apply_action(actions[chosen]);
                node_value = child.nerual_network_recursion( player_traverser, rng, regret_gamstate_training, regret_training, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_regret_samples_colected, C, hand_lookup_table, N_stratagy_samples_colected, stratagy_gamstate_training, regret_N_vec, stratagy_N_vec,t, strategy_capacity, resivour_rng);
            }
            return node_value;
        }
    }

    //####################################################################################################
    //####################################################################################################
    //####################################################################################################



 







   static int hand_evaluation(const std::vector<float>& hand, int ranks, const std::vector <uint32_t>& strait_lookup, const long long P5, const long long P4, const long long P3, const long long P2, const long long P1) {
        //straight flush
        //quads
        //full house
        //flush
        //straight
        //trips
        //two pair
        //pair
        //high


        std::vector rank_count(ranks, 0);
        std::vector suit_count(4, 0);
        for (int i = 0; i < hand.size(); i++) {
            int card = static_cast<int>(hand[i]);
            int suit = card % 4;
            int rank = card / 4;
            suit_count[suit] += 1;
            rank_count[rank] += 1;
        }
        uint32_t strait_mask = 0;
        for (int i = 0; i < ranks; i++) {
            if (rank_count[i] != 0) {
                strait_mask |= (1 << i);
            }
        }
        bool is_strait = false;
        int strait_rank;
        for (int i = 0; i < strait_lookup.size(); i++) {
            if (strait_mask == strait_lookup[i]) {
                is_strait = true;
                if (i < 9) {
                    strait_rank = i + 4;
                }
                else {
                    strait_rank = 3;
                }
                break;
            }
        }
        bool is_flush = false;
        for (int i = 0; i < suit_count.size(); i++) {
            if (5 == suit_count[i]) {
                is_flush = true;
                break;
            }
        }
        if (is_flush && is_strait) {
            return 8 * P5 + strait_rank * P4;
        }
        else {
            bool is_quad = false;
            bool is_full_house = false;
            //flush
            //straight
            bool is_trips = false;
            bool is_two_pair = false;
            bool is_pair = false;

            int quad_rank;
            int trip_rank;
            int pair_rank;
            int two_pair_rank;
            for (int i = 0; i < ranks; i++) {
                if (rank_count[i] == 4) {
                    is_quad = true;
                    quad_rank = i;
                    break;
                }
                if (rank_count[i] == 3) {
                    is_trips = true;
                    trip_rank = i;
                }
                if (rank_count[i] == 2) {
                    if (is_pair) {
                        is_two_pair = true;
                        two_pair_rank = i;
                    }
                    else {
                        is_pair = true;
                        pair_rank = i;
                    }
                }
            }
            if (is_pair && is_trips) {
                is_full_house = true;
            }
            if (is_quad) {
                int spare_0;
                for (int i = ranks - 1; i > -1; i--) {
                    if (rank_count[i] != 0 && rank_count[i] != 4) {
                        spare_0 = i;
                        break;
                    }
                }
                return 7*P5 + quad_rank*P4 + spare_0*P3;
            }
            else if (is_full_house) {
                return 6 * P5 + trip_rank * P4 + pair_rank * P3;
            }
            else if (is_flush) {
                std::vector<int> kickers;
                for (int i = 0; i < ranks; i++) {
                    if (rank_count[i] != 0) {
                        kickers.push_back(i);
                    }

                }
                return 5 * P5 + kickers[4] * P4 + kickers[3] * P3 + kickers[2] * P2 + kickers[1] * P1 + kickers[0];
            }
            else if (is_strait) {
                return 4 * P5 + strait_rank * P4;
            }
            else if (is_trips) {
                std::vector<int> kickers;
                for (int i = 0; i < ranks; i++) {
                    if (rank_count[i] != 0 && rank_count[i] != 3) {
                        kickers.push_back(i);
                    }

                }
                return 3 * P5 + trip_rank * P4 + kickers[1] * P3 + kickers[0];
            }
            else if (is_two_pair) {
                int kicker3;
                for (int i = 0; i < ranks; i++) {
                    if (rank_count[i] != 0 && rank_count[i] != 2) {
                        kicker3 = i;
                    }
                }
                return 2 * P5 + two_pair_rank * P4 + pair_rank * P3 + kicker3 * P2;
            }
            else if (is_pair) {
                std::vector<int> kickers;
                for (int i = 0; i < ranks; i++) {
                    if (rank_count[i] != 0 and rank_count[i] != 2) {
                        kickers.push_back(i);
                    }

                }
                return 1 * P5 + pair_rank * P4 + kickers[2] * P3 + kickers[1] * P2 + kickers[0] * P1;
            }
            else {
                std::vector<int> kickers;
                for (int i = 0; i < ranks; i++) {
                    if (rank_count[i] != 0) {
                        kickers.push_back(i);
                    }

                }
                return kickers[4] * P4 + kickers[3] * P3 + kickers[2] * P2 + kickers[1] * P1 + kickers[0];
            }//hight card
        }
    }

    //#########################################################################################################
    //#########################################################################################################
    //#########################################################################################################

    static void possible_hands_calc(int predetor, int ranks, const std::vector<float>& cards, std::vector<std::vector<float>>& cards_adress) {
        int predetor_card_1 = cards[predetor * 2];
        int predetor_card_2 = cards[predetor * 2 + 1];
        int unique_cards = ranks * 4;
        for (int i = 0; i < unique_cards; i++) {
            if (i != predetor_card_1 && i != predetor_card_2) {
                for (int j = i + 1; j < unique_cards; j++) {
                    if (j != predetor_card_1 && j != predetor_card_2) {
                        cards_adress.push_back({ (float)i,(float)j });
                    }
                }
            }
        }
    }

    static std::vector<std::vector<float>> flop_hands_calc(int predetor, int ranks, const std::vector<int>& predetor_cards) {
        int predetor_card_1 = predetor_cards[0];
        int predetor_card_2 = predetor_cards[1];
        std::vector<std::vector<float>>flops;
        int unique_cards = ranks * 4;
        for (int i = 0; i < unique_cards; i++) {
            if (i != predetor_card_1 && i != predetor_card_2) {
                for (int j = i + 1; j < unique_cards; j++) {
                    if (j != predetor_card_1 && j != predetor_card_2) {
                        for (int k = j + 1; k < unique_cards; k++) {
                            if (k != predetor_card_1 && k != predetor_card_2) {
                                flops.push_back({ (float)i,(float)j,(float)k });
                            }
                        }
                    }
                }
            }
        }
        return flops;
    }

    std::vector<float> exploitablity_terminal_returns(int predetor, int predetor_score) {
        std::vector<float> returns(player_contributions_.size(), 0.0f);
        int player_score = wining_card(1 - predetor);
        if (predetor_score > player_score) {
            returns[predetor] += pot_/2;
            returns[1- predetor] -= pot_ / 2;
        }
        else if (predetor_score < player_score) {
            returns[1- predetor] += pot_/2;
            returns[predetor] -= pot_ / 2;
        }
        return returns;
    }


    std::vector<float> river_hands_calc(int predetor, int ranks, const std::vector<int>& predetor_cards, int round) {
        int predetor_card_1 = predetor_cards[0];
        int predetor_card_2 = predetor_cards[1];
        int public1 = cards_[n_players_ * 2];
        int public2 = cards_[n_players_ * 2 + 1];
        int public3 = cards_[n_players_ * 2 + 2];
        int public4 = cards_[n_players_ * 2 + 3];
        std::vector<float>rivers;
        int unique_cards = ranks * 4;
        for (int i = 0; i < unique_cards; i++) {
            if (i != predetor_card_1 && i != predetor_card_2 && i != public1 && i != public2 && i != public3 && (round != 3 || i != public4)) {
                rivers.push_back(i);
            }
        }
        return rivers;
    }


    std::string predetor_get_key(const std::vector<float>& player_cards, int predetor) {
        std::vector<float>cards = cards_;
        cards[(1 - predetor) * 2] = player_cards[0];
        cards[(1 - predetor) * 2 + 1] = player_cards[1];
        std::string key;
        if (cards[2 * player_turn_] < cards[2 * player_turn_ + 1]) {
            key += std::to_string((int)cards[2 * player_turn_]) + "|Player" + std::to_string((int)cards[2 * player_turn_ + 1]);
        }
        else {
            key += std::to_string((int)cards[2 * player_turn_ + 1]) + "|Player" + std::to_string((int)cards[2 * player_turn_]);
        }
        if (round_ >= 2) {
            std::vector<float> flop(cards.begin() + n_players_ * 2, cards.begin() + n_players_ * 2 + 3);
            std::sort(flop.begin(), flop.end());
            for (int i = 0; i < 3; i++) {
                key += "|Public" + std::to_string((int)flop[i]);
            }
            for (int i = 3; i < round_ + 1; i++) {
                key += "|Public" + std::to_string((int)cards[2 * n_players_ + i]);
            }

        }
        for (std::string& a : history_) key += "|" + a;
        return key;
    }




    std::vector<float> exploitiblity_calc( int predator, std::vector<int>const& predetor_cards, std::vector<std::vector<float>>const& possible_hands, std::vector<float>const& weight_vec, int ranks,  std::vector<std::vector<float>>const& flops, std::vector<Matrix>& WN_stratagy, int hidden_layer_size, int n_hidden_layers, float max_pot, float max_player_contribution, int C ,const std::unordered_map<int, float>& hand_lookup_table, std::unordered_map<std::string, std::vector<float>>& strat_cache) {
        std::vector<std::string> actions = legal_actions();
        if (actions.empty()) {
            if (n_seats_() != 1 && round_ <max_rounds_) {
                std::vector<float> agraget_return = { 0,0 };
                float a0 = 0;
                float a1 = 0;
                if (round_ == 1) {
                    for (int i = 0; i < flops.size(); i ++) {
                        Decition_Node sibling = *this;
                        sibling.player_traversing_ = predator;
                        for (int j = 0; j < 3; j++) {
                            sibling.cards_[n_players_ * 2 + j] = flops[i][j];
                        }
                        std::vector<float> new_weight_vec = weight_vec;
                        Decition_Node sibling2 = sibling.new_round();
                        int a = flops[i][0];
                        int b = flops[i][1];
                        int c = flops[i][2];
                        for (int k = 0; k < possible_hands.size(); k++) {
                            int card1 = possible_hands[k][0];
                            int card2 = possible_hands[k][1];
                            if (card1 == a || card1 == b || card1 == c || card2 == a || card2 == b || card2 == c) {
                                new_weight_vec[k] = 0;
                            }
                        }
                        std::vector<float> exp_return = sibling2.exploitiblity_calc( predator, predetor_cards, possible_hands, new_weight_vec, ranks, flops, WN_stratagy,hidden_layer_size, n_hidden_layers, max_pot,  max_player_contribution,  C, hand_lookup_table, strat_cache);
                        a0 += exp_return[0];
                        a1 += exp_return[1];
                    }
                    int c = (4 * ranks - 4)* (4 * ranks - 5)* (4 * ranks - 6) / 6;
                
                    agraget_return[0] = a0 / c;
                    agraget_return[1] = a1 / c;
                    
                }
                else if (round_ == 2 || round_ == 3) {
                    std::vector<float> rivers = river_hands_calc(predator,ranks, predetor_cards, round_);
                    for (int i = 0; i < rivers.size(); i++) {
                        Decition_Node sibling = *this;
                        sibling.player_traversing_ = predator;
                        sibling.cards_[n_players_ * 2 + round_+1] = rivers[i];
                        std::vector<float> new_weight_vec = weight_vec;
                        Decition_Node sibling2 = sibling.new_round();
                        int a = rivers[i];
                        for (int k = 0; k < possible_hands.size(); k++) {
                            int card1 = possible_hands[k][0];
                            int card2 = possible_hands[k][1];
                            if (card1 == a ||  card2 == a ) {
                                new_weight_vec[k] = 0;
                            }
                        }
                        std::vector<float> exp_return = sibling2.exploitiblity_calc( predator, predetor_cards, possible_hands, new_weight_vec, ranks, flops, WN_stratagy, hidden_layer_size, n_hidden_layers, max_pot, max_player_contribution, C, hand_lookup_table, strat_cache);
                        a0 += exp_return[0];
                        a1 += exp_return[1];
                    }
                    
                    int d0 = rivers.size()-2;
                    agraget_return[0] = a0 / d0;
                    agraget_return[1] = a1 / d0;
                }
                return agraget_return;
            }
            else {
                std::vector<float> node_value(n_players_, 0.0f);
                


                int n_seats = std::count(player_status_.begin(), player_status_.end(), "playing");
                if (n_seats == 1) {
                    std::vector<float> v(2, 0);
                    auto it = std::find(player_status_.begin(), player_status_.end(), "playing");
                    int winner_idx = std::distance(player_status_.begin(), it);
                    for (int i = 0; i < n_players_; i++) {
                        if (i == winner_idx) {
                            v[i] = pot_ - player_contributions_[i];
                        }
                        else {
                            v[i] = -1 * player_contributions_[i];
                        }
                    }
                    for (int j = 0; j < possible_hands.size(); j++) {
                        for (int k = 0; k < n_players_; k++) {
                            node_value[k] += v[k] * weight_vec[j];
                        }
                    }
                }
                else {
                    int predetor_score = wining_card(predator);
                    int opponent = 1 - predator;    
                    float c1 = cards_[opponent * 2];
                    float c2 = cards_[opponent * 2 + 1];
                    for (int j = 0; j < possible_hands.size(); j++) {
                        if (weight_vec[j] != 0) {
                            cards_[opponent * 2] = possible_hands[j][0];
                            cards_[opponent * 2 + 1] = possible_hands[j][1];
                            std::vector<float> v = exploitablity_terminal_returns(predator, predetor_score);
                            for (int k = 0; k < n_players_; k++) {
                                node_value[k] += v[k] * weight_vec[j];
                            }
                        }
                    }
                    cards_[opponent * 2] = c1;
                    cards_[opponent * 2 + 1] = c2;
                }
                return node_value;
            }
        }
        else {
            std::vector<float> node_value(n_players_, 0.0f);
            float max_return = -1e9;
            std::vector<std::vector<float>> opponent_strategy(possible_hands.size());
            
            if (player_turn_ != predator){
                std::string keystart;
                for (auto& a : history_) keystart += (a != "check" ? a[0] : 'k');                 // f/c/k(check)/b/r: one char per action
                keystart += '|';
                for (int b = 0; b < (round_ >= 2 ? round_ + 1 : 0); b++)
                    keystart += std::to_string((int)cards_[n_players_ * 2 + b]) + ',';
                keystart += '|';
                for (int j = 0; j < possible_hands.size(); j++) {
                    std::string key = keystart;
                    key +=  std::to_string((int)possible_hands[j][0]) + ',' + std::to_string((int)possible_hands[j][1]);

                    auto it = strat_cache.find(key);
                    if (it != strat_cache.end()) {
                        opponent_strategy[j] = it->second;
                    }
                    else {
                        if (weight_vec[j] != 0) {
                            Decition_Node sibling = *this;
                            sibling.cards_[player_turn_ * 2] = possible_hands[j][0];
                            sibling.cards_[player_turn_ * 2 + 1] = possible_hands[j][1];
                            std::vector<float> pcg = sibling.player_centric_gamstate(player_turn_);
                            std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, pcg, C, hand_lookup_table, max_rounds_);
                            input.push_back(1.0f);
                            std::vector<float> j_stratagy = forward_pass(input, WN_stratagy, hidden_layer_size, n_hidden_layers, max_pot);

                            std::vector<float> j_stratagy_normalised(actions.size(), 0);
                            for (int i = 0; i < actions.size(); i++) {
                                float r = j_stratagy[action_index_.at(actions[i])];
                                j_stratagy_normalised[i] = (r > 0.0f) ? r : 0.0f;
                            }

                            float total_regret = 0;
                            for (int k = 0; k < j_stratagy_normalised.size(); k++) {
                                total_regret += j_stratagy_normalised[k];
                            }

                            if (total_regret != 0) {
                                for (int l = 0; l < j_stratagy_normalised.size(); l++) {
                                    j_stratagy_normalised[l] /= total_regret;
                                }
                            }
                            else {
                                for (int x = 0; x < j_stratagy_normalised.size(); x++) {
                                    j_stratagy_normalised[x] = 1.0f / actions.size();
                                }
                            }

                            opponent_strategy[j] = j_stratagy_normalised;
                            strat_cache.emplace(std::move(key), j_stratagy_normalised);
                        }
                    }
                }
            }
            


            for (int i = 0; i < actions.size(); i++) {

                if (player_turn_ == predator) {
                    Decition_Node child = apply_action(actions[i]);
                    std::vector<float> v = child.exploitiblity_calc( predator, predetor_cards, possible_hands, weight_vec, ranks, flops, WN_stratagy, hidden_layer_size, n_hidden_layers, max_pot, max_player_contribution, C, hand_lookup_table, strat_cache);
                    if (v[predator] > max_return) {
                        node_value = v;
                        max_return = v[predator];
                    }
                }
                else {
                    Decition_Node root = apply_action(actions[i]);
                    std::vector<float> new_weight_vec = weight_vec;
                    for (int j = 0; j < possible_hands.size(); j++) {
                        if (weight_vec[j] != 0) {
                            new_weight_vec[j] *= opponent_strategy[j][i];
                            }
                        }
                    std::vector<float> v = root.exploitiblity_calc( predator, predetor_cards, possible_hands, new_weight_vec, ranks, flops, WN_stratagy, hidden_layer_size, n_hidden_layers, max_pot, max_player_contribution, C, hand_lookup_table, strat_cache);
                    for (int k = 0; k < n_players_; k++) {
                        node_value[k] += v[k];
                    }
                }
            }
            return node_value;
        }
        
    }   




    std::vector<float> handresponse( int ranks, std::vector<Matrix>& WN_stratagy, int hidden_layer_size, int n_hidden_layers, float max_pot, float max_player_contribution, int C, const std::unordered_map<int, float>& hand_lookup_table) {
        
        std::vector<std::string> actions = legal_actions();
        std::vector<float> pcg = player_centric_gamstate(player_turn_);
        std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, pcg, C, hand_lookup_table, max_rounds_);
        input.push_back(1.0f);

        std::vector<float> j_stratagy = forward_pass(input, WN_stratagy, hidden_layer_size, n_hidden_layers, max_pot);

        std::vector<float> j_stratagy_normalised(5, 0);
        for (int i = 0; i < actions.size(); i++) {
            float r = j_stratagy[action_index_.at(actions[i])];
            j_stratagy_normalised[action_index_.at(actions[i])] = (r > 0.0f) ? r : 0.0f;
        }

        float total_regret = 0;
        for (int k = 0; k < j_stratagy_normalised.size(); k++) {
            total_regret += j_stratagy_normalised[k];
        }

        if (total_regret != 0) {
            for (int l = 0; l < j_stratagy_normalised.size(); l++) {
                j_stratagy_normalised[l] /= total_regret;
            }
        }
        else {
            for (int x = 0; x < actions.size(); x++) {
                j_stratagy_normalised[action_index_.at(actions[x])] = 1.0f / actions.size();
            }
        }
        return j_stratagy_normalised;

    }
};