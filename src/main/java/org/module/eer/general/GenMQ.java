package org.module.eer.general;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.module.eer.model.Entity;
import org.module.eer.model.Relationship;

public class GenMQ {

	public static double apply(GenState state) {
		Map<Set<Byte>, Pair> linkMap = new HashMap<>(state.modules.size());
		for (Set<Byte> s : state.modules) {
			linkMap.put(s, new Pair());
		}
		for (Relationship r : state.relations) {
			Entity entityA = r.a;
			Entity entityB = r.b;
			Set<Byte> setR = null;
			Set<Byte> setA = null;
			Set<Byte> setB = null;
			byte indexR = state.indices.get(r);
			byte indexA = state.indices.get(entityA);
			byte indexB = state.indices.get(entityB);
			for (Set<Byte> s : state.modules) {
				if (s.contains(indexR)) {
					setR = s;
				}
				if (s.contains(indexA)) {
					setA = s;
				}
				if (s.contains(indexB)) {
					setB = s;
				}
			}
			if (setA == setR) {
				linkMap.get(setA).intraLink++;
			}else {
				linkMap.get(setA).interLink++;
				linkMap.get(setR).interLink++;
			}
			if (setB == setR) {
				linkMap.get(setB).intraLink++;
			}else {
				linkMap.get(setB).interLink++;
				linkMap.get(setR).interLink++;
			}
		}
		double sum = 0;
		for(Entry<Set<Byte>, Pair> e: linkMap.entrySet()) {
			int intraLinks = e.getValue().intraLink;
			int interLinks = e.getValue().interLink;
			if (intraLinks > 0) {
				sum += intraLinks / (intraLinks + interLinks / 2.0);
			}
		}
		return sum;
	}
}

class Pair {
	// inside link
	int intraLink;
	// outside link
	int interLink;

	public Pair() {
		intraLink = 0;
		interLink = 0;
	}

}