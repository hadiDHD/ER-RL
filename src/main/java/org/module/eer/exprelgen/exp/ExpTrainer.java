package org.module.eer.exprelgen.exp;

import com.google.gson.Gson;
import kotlin.Pair;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.module.eer.general.GenSolver;
import org.module.eer.model.ERModel;
import org.module.eer.model.Element;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class ExpTrainer {

    static int epochStep = 1000;
    static int epochNumber = 1;

    static QLearningConfiguration QL =
            QLearningConfiguration.builder()
                    .maxEpochStep(epochStep)
                    .maxStep(10000)
                    .gamma(0)
                    .rewardFactor(1)
                    .build();

    static DQNDenseNetworkConfiguration NET =
            DQNDenseNetworkConfiguration.builder()
                    .numLayers(2)
                    .numHiddenNodes(ExpState.OUTPUT_SIZE).
                    build();

    public static void main(String[] args) {
        try {
            // create Gson instance
            Gson gson = new Gson();

            JFileChooser chooser = new JFileChooser();

            chooser.showOpenDialog(null);

            // create a reader
            Reader reader = Files.newBufferedReader(Paths.get(chooser.getSelectedFile().getAbsolutePath()));

            // convert JSON string to User object
            ERModel model = gson.fromJson(reader, ERModel.class);
            model.init();

            // close reader
            reader.close();
            ExpMDP mdp = new ExpMDP(model);
            //TODO
//            DQNPolicy<GenState> policy = DQNPolicy.load("C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\exp\\1649252901947");
//            QLearningDiscreteDense<ExpState> qLearning = new QLearningDiscreteDense<>(mdp, policy.getNeuralNet(), QL);
            QLearningDiscreteDense<ExpState> qLearning = new QLearningDiscreteDense<>(mdp, NET, QL);
            qLearning.train();

            int validActionRange = ExpState.OUTPUT_SIZE;
            play(qLearning.getPolicy(), mdp, validActionRange);

            File nn = new File("neural networks/exp/" + System.currentTimeMillis());
            System.out.println(nn.getAbsolutePath());
            try {
                nn.createNewFile();
                qLearning.getPolicy().save(nn.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static double play(DQNPolicy policy, ExpMDP mdp, int actionRange) {
        LegacyMDPWrapper<ExpState, Integer, DiscreteSpace> mdpWrapper = new LegacyMDPWrapper<>(mdp, null);

        Learning.InitMdp<Observation> initMdp = GenSolver.refacInitMdp(mdpWrapper, null);
        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        int lastAction = mdpWrapper.getActionSpace().noOp();
        int action;

        ExpState bestState = (ExpState) mdp.state.dup();
        int goodActions = 0;

        while (!mdpWrapper.isDone()) {

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
            if (maxActionIndex == -1) {
                break;
            }

//            System.out.println("actions: " + Arrays.toString(actions));
//            System.out.println("invalids: " + Arrays.toString(invalidActions.toArray()));
            Pair<Integer, Integer> p = ExpState.getTriangularMatrixRowAndColumn(maxActionIndex);
            Element eI = mdp.state.er.elements.get(p.getFirst());
            Element eJ = mdp.state.er.elements.get(p.getSecond());
//            System.out.println("move: " + eI + " to " + eJ);

            StepReply<Observation> stepReply = mdpWrapper.step(action);
            reward += stepReply.getReward();

//            System.out.println("Reward: " + reward);

            obs = stepReply.getObservation();

            if (mdp.state.reward > bestState.reward) {
                bestState = (ExpState) mdp.state.dup();
//                System.out.println("YEEEEEEEEEEEEEEEEEEES!!!!");
                goodActions++;
            }
//            if(mdp.state.modules.size() < 8){
//                mdp.state.print();
//            }
        }

//        System.out.println(ExpMQ.apply(bestState));
        bestState.print();
        System.out.println("good actions: " + goodActions);
        return reward;
    }
}

