package org.module.eer.general;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.module.eer.model.Element;
import org.module.eer.model.Entity;
import org.module.eer.model.Relationship;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class GenSolver {
    public static void main(String[] args) {
        try {
            DQNPolicy<GenState> policy = DQNPolicy.load("C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\1648567356166");
            test(policy);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void test(DQNPolicy<GenState> policy) {
        Entity e1 = new Entity("E1");
        Entity e3 = new Entity("E3");
        Entity e5 = new Entity("E5");
        Relationship r2 = new Relationship("R2",e1, e3);
        Relationship r4 = new Relationship("R4", e3, e5);
        ArrayList<Element> elements = new ArrayList<>();
        elements.add(e1);
        elements.add(e3);
        elements.add(e5);
        elements.add(r2);
        elements.add(r4);
        ArrayList<Relationship> relationships = new ArrayList<>();
        relationships.add(r2);
        relationships.add(r4);
        HashMap<Element, Byte> indices = new HashMap<>();
        indices.put(e1, (byte) 1);
        indices.put(e3, (byte) 2);
        indices.put(e5, (byte) 3);
        indices.put(r2, (byte) 4);
        indices.put(r4, (byte) 5);
        GenMDP mdp = new GenMDP(indices, elements, relationships);
        int validActionRange = elements.size() * (elements.size() - 1) / 2;
        play(policy, mdp, validActionRange);
    }

    public static double play(DQNPolicy policy, GenMDP mdp, int actionRange) {
        LegacyMDPWrapper<GenState, Integer, DiscreteSpace> mdpWrapper = new LegacyMDPWrapper<>(mdp, null);

        Learning.InitMdp<Observation> initMdp = refacInitMdp(mdpWrapper, null);
        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        int lastAction = mdpWrapper.getActionSpace().noOp();
        int action;

        GenState bestState = (GenState) mdp.state.dup();

        while (!mdpWrapper.isDone()) {

            if (obs.isSkipped()) {
                action = lastAction;
            } else {
                INDArray output = policy.getNeuralNet().output(obs);
                double[] actions = output.toDoubleVector();
                List<Integer> invalidActions = mdp.state.getInvalidActions();
                for (int ac : invalidActions) {
                    actions[ac] = 0.0;
                }
                double maxActionValue = Double.MIN_VALUE;
                int maxActionIndex = -1;
                for (int i = 0; i < actionRange; i++) {
                    if (actions[i] > maxActionValue) {
                        maxActionValue = actions[i];
                        maxActionIndex = i;
                    }
                }
                action = maxActionIndex;
            }

            lastAction = action;

            StepReply<Observation> stepReply = mdpWrapper.step(action);
            reward += stepReply.getReward();

            obs = stepReply.getObservation();

            if (mdp.state.accReward > bestState.accReward) {
                bestState = (GenState) mdp.state.dup();
            }
        }

        GenMQ.apply(bestState);
        bestState.print();
        System.out.println(reward);
        return reward;
    }

    public static <O extends Encodable, AS extends ActionSpace<Integer>> Learning.InitMdp<Observation> refacInitMdp(LegacyMDPWrapper<O, Integer, AS> mdpWrapper, IHistoryProcessor hp) {

        double reward = 0;

        Observation observation = mdpWrapper.reset();

        int action = mdpWrapper.getActionSpace().noOp(); //by convention should be the NO_OP
        while (observation.isSkipped() && !mdpWrapper.isDone()) {

            StepReply<Observation> stepReply = mdpWrapper.step(action);

            reward += stepReply.getReward();
            observation = stepReply.getObservation();

        }

        return new Learning.InitMdp(0, observation, reward);
    }
}
