package org.module.eer.expmod.exp;

import org.deeplearning4j.rl4j.learning.async.IAsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CThreadDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.AdvantageActorCriticUpdateAlgorithm;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

import org.deeplearning4j.rl4j.learning.async.*;
import org.deeplearning4j.rl4j.learning.async.*;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.rng.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 * <p>
 * Local thread as described in the https://arxiv.org/abs/1602.01783 paper.
 */
public class MyA3CThreadDiscrete extends MyAsyncThreadDiscrete<ExpState, IActorCritic> {

    final public A3CLearningConfiguration configuration;
    final public IAsyncGlobal<IActorCritic> asyncGlobal;
    final public int threadNumber;

    final public Random rnd;

    public MyA3CThreadDiscrete(MDP<ExpState, Integer, DiscreteSpace> mdp, IAsyncGlobal<IActorCritic> asyncGlobal,
                             A3CLearningConfiguration a3cc, int deviceNum, TrainingListenerList listeners,
                             int threadNumber) {
        super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
        this.configuration = a3cc;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;

        Long seed = configuration.getSeed();
        rnd = Nd4j.getRandom();
        if (seed != null) {
            rnd.setSeed(seed + threadNumber);
        }

        setUpdateAlgorithm(buildUpdateAlgorithm());
    }

    @Override
    public IActorCritic getCurrent() {
        return current;
    }

    @Override
    protected IAsyncGlobal<IActorCritic> getAsyncGlobal() {
        return asyncGlobal;
    }

    @Override
    protected IAsyncLearningConfiguration getConfiguration() {
        return configuration;
    }

    @Override
    protected Policy<Integer> getPolicy(IActorCritic net) {
        return new MyPolicy(net, rnd);
    }

    /**
     * calc the gradients based on the n-step rewards
     */
    @Override
    protected UpdateAlgorithm<IActorCritic> buildUpdateAlgorithm() {
        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape() : getHistoryProcessor().getConf().getShape();
        return new AdvantageActorCriticUpdateAlgorithm(asyncGlobal.getTarget().isRecurrent(), shape, getMdp().getActionSpace().getSize(), configuration.getGamma());
    }
}
