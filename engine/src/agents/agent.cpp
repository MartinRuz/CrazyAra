/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: agent.cpp
 * Created on 17.06.2019
 * @author: queensgambit
 */

#include <iostream>
#include <chrono>

#include "agent.h"
#include "../util/communication.h"
#include "../util/blazeutil.h"
#include "stateobj.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace crazyara;


void Agent::set_best_move(size_t moveCounter)
{
    if (moveCounter < playSettings->temperatureMoves && playSettings->initTemperature > 0.01) {
        info_string("Sample move");
        DynamicVector<double> policyProbSmall = evalInfo->policyProbSmall;
        apply_temperature(policyProbSmall, get_current_temperature(*playSettings, moveCounter));
        if (playSettings->quantileClipping != 0) {
            apply_quantile_clipping(playSettings->quantileClipping, policyProbSmall);
        }
        size_t moveIdx = random_choice(policyProbSmall);
        evalInfo->bestMove = evalInfo->legalMoves[moveIdx];
    }
    else {
        evalInfo->bestMove = evalInfo->pv[0][0];
    }
}

Agent::Agent(NeuralNetAPI* net, PlaySettings* playSettings, bool verbose):
    NeuralNetAPIUser(net),
    playSettings(playSettings), verbose(verbose), isRunning(false)
{
}

void Agent::set_search_settings(StateObj *pos, SearchLimits *searchLimits, EvalInfo* evalInfo)
{
    this->state = pos;
    this->searchLimits = searchLimits;
    this->evalInfo = evalInfo;
}

Action Agent::get_best_action()
{
    return evalInfo->bestMove;
}

void Agent::lock()
{
    runnerMutex.lock();
}

void Agent::unlock()
{
    runnerMutex.unlock();
}

void Agent::perform_action()
{
    isRunning = true;
    evalInfo->start = chrono::steady_clock::now();
    this->evaluate_board_state();
    evalInfo->end = chrono::steady_clock::now();
    set_best_move(state->steps_from_null());
    info_msg(*evalInfo);
    info_string(state->fen());
    #ifdef MODE_STRATEGO
        info_bestmove(StateConstants::action_to_uci(evalInfo->bestMove, state->is_chess960()) + " equals " + state->action_to_string(evalInfo->bestMove));
    #else
        info_bestmove(StateConstants::action_to_uci(evalInfo->bestMove, state->is_chess960()));
    #endif
    isRunning = false;
    runnerMutex.unlock();
}
/*
void Agent::print_debug_file(const StateObj* state, const vector<size_t>& customOrdering, const SearchSettings* searchSettings)
{
    const string header = "  #  | Move  |    first variance    |  second variance |  final variance  |  visits  |   min selection    |    max selection   |   final term   |   policy   |    ";
    const string filler = "-----+-------+----------------------+------------------+------------------+----------+--------------------+--------------------+----------------+------------+";
    ofstream outfile;
    outfile.open("debug.txt", std::ios_base::app);
    outfile << header << endl
        << std::showpoint << std::fixed << std::setprecision(7)
        << filler << endl;
    for (int idx = 0; idx < evalInfo->legalMoves.size(); idx++) {
        const size_t childIdx = customOrdering.size() == evalInfo->legalMoves.size() ? customOrdering[idx] : idx;
        const Action move = evalInfo->legalMoves[childIdx];
        outfile << " " << setfill('0') << setw(3) << childIdx << " | " << setfill(' ');
        if (state == nullptr) {
            outfile << setw(5) << StateConstants::action_to_uci(move, false) << " | ";
        }
        else {
            outfile << setw(5) << state->action_to_san(move, evalInfo->legalMoves, false, false) << " | ";
        }
        //DynamicVector<float> u_values = get_current_u_values(searchSettings);
        outfile << setw(20) << d->stdev_one[childIdx] << " | ";
        outfile << setw(16) << d->stdev_two[childIdx] << " | ";
        outfile << setw(16) << d->stdDev[childIdx] << " | ";
        outfile << setw(8) << d->childNumberVisits[childIdx] << " | ";
        outfile << setw(18) << d->min_term[childIdx] << " | ";
        outfile << setw(18) << d->max_term[childIdx] << " | ";
        //outfile << setw(14) << u_values[childIdx] << " | " << endl;
    }
    outfile.close();
}*/

void run_agent_thread(Agent* agent)
{
    agent->perform_action();
    // inform the agent of the move, so the tree can potentially be reused later
    agent->apply_move_to_tree(agent->get_best_action(), true);
}

void apply_quantile_clipping(float quantile, DynamicVector<double>& policyProbSmall)
{
    double thresh = get_quantile(policyProbSmall, quantile);
    for (size_t idx = 0; idx < policyProbSmall.size(); ++idx) {
        if (policyProbSmall[idx] < thresh) {
            policyProbSmall[idx] = 0;
        }
    }
    policyProbSmall /= sum(policyProbSmall);
}
