package org.module.eer.exprelindimove.exp;

import com.google.gson.Gson;
import kotlin.Pair;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.module.eer.general.GenSolver;
import org.module.eer.model.ERModel;
import org.module.eer.model.Element;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class ExpTrainerStagedA3C {

    static int epochStep = ExpState.MAX_ELEMENTS;


    private static final ActorCriticDenseNetworkConfiguration NET_A3C = ActorCriticDenseNetworkConfiguration.builder()
            .numLayers(3)
            .numHiddenNodes(ExpState.OUTPUT_SIZE)
            .learningRate(0.001)
            .updater(new Adam(0.001))
            .l2(0.000)
            .build();

    public static void main(String[] args) {
        try {
            System.setProperty("org.bytedeco.openblas.load", "mkl");
            // create Gson instance
            Gson gson = new Gson();

            JFileChooser chooser = new JFileChooser();

            chooser.showOpenDialog(null);

            // create a reader
            Reader reader = Files.newBufferedReader(Paths.get(chooser.getSelectedFile().getAbsolutePath()));

            // convert JSON string to User object
            ERModel model = gson.fromJson(reader, ERModel.class);
            model.init();
            System.out.println(model.elements.size());

            // close reader
            reader.close();
            ExpMDP mdp = new ExpMDP(model);
            ACPolicy<ExpState> policy = null;
//            String loadName = "1649933246107";
//            ACPolicy<ExpState> policy = ACPolicy.load(
//                    "C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\indimove\\a3c\\" + loadName + "_value",
//                    "C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\indimove\\a3c\\" + loadName + "_policy"
//            );
            A3CDiscreteDense<ExpState> dql;
            for (int i = 0; i < 100; i++) {
                A3CLearningConfiguration LEARNING_A3C =
                        A3CLearningConfiguration.builder()
                                .maxEpochStep(epochStep)
                                .maxStep(10000)
                                .numThreads(32)
                                .rewardFactor(0.01)
                                .gamma(0.99)
                                .nStep(5)
                                .build();
                if (policy == null) {
                    dql = new A3CDiscreteDense<>(mdp, NET_A3C, LEARNING_A3C);
                } else {
                    dql = new A3CDiscreteDense<>(mdp, policy.getNeuralNet(), LEARNING_A3C);
                }
                dql.train();
                policy = dql.getPolicy();

                int validActionRange = ExpState.OUTPUT_SIZE;
                double reward = play(dql.getPolicy(), mdp, validActionRange);

                if (i % 10 == 9) {
                    try {
                        String name = System.currentTimeMillis() + "";
                        File valueN = new File("neural networks/indimove/a3c/" + name + "_value");
                        File policyN = new File("neural networks/indimove/a3c/" + name + "_policy");
                        System.out.println(valueN.getAbsolutePath());
//                        nn.createNewFile();
                        policy.save(valueN.getAbsolutePath(), policyN.getAbsolutePath());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static double play(ACPolicy policy, ExpMDP mdp, int actionRange) {
        LegacyMDPWrapper<ExpState, Integer, DiscreteSpace> mdpWrapper = new LegacyMDPWrapper<>(mdp, null);

        Learning.InitMdp<Observation> initMdp = GenSolver.refacInitMdp(mdpWrapper, null);
        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        int lastAction = mdpWrapper.getActionSpace().noOp();
        int action;

        ExpState bestState = (ExpState) mdp.state.dup();
        int goodActions = 0;

        while (!mdpWrapper.isDone()) {

            INDArray output = policy.getNeuralNet().outputAll(obs.getData())[1];
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

            if (mdp.state.mq > bestState.mq) {
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

