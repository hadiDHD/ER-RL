package org.module.eer.expmod.simp;

import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;

public class MyPolicy extends ACPolicy<ExpState> {

    private IActorCritic actorCritic;
    Random rnd;

    public MyPolicy(IActorCritic actorCritic) {
        super(actorCritic);
        this.actorCritic = actorCritic;
    }

    public MyPolicy(IActorCritic actorCritic, Random rnd) {
        super(actorCritic, rnd);
        this.actorCritic = actorCritic;
        this.rnd = rnd;
    }

    public Integer nextAction(Observation obs, Integer validActionRange) {
        return nextAction(obs.getData(), validActionRange);
    }

    private Integer nextAction(INDArray input, Integer validActionRange) {
        INDArray output = actorCritic.outputAll(input)[1];
        if (rnd == null) {
            return Learning.getMaxAction(output);
        }
        float rVal = rnd.nextFloat();
        double wastedSum = 0;
        for (int i = validActionRange; i < output.length(); i++) {
            wastedSum += output.getFloat(i);
        }
        float excess = (float) (wastedSum / validActionRange);
        for (int i = 0; i < validActionRange; i++) {
            //System.out.println(i + " " + rVal + " " + output.getFloat(i));
            if (rVal < (output.getFloat(i) + excess)) {
                return i;
            } else
                rVal -= (output.getFloat(i) + excess);
        }
        System.out.println("validActionRange: " + validActionRange);
        throw new RuntimeException("Output from network is not a probability distribution: " + output);
    }
}
