UE22 Projet d'informatique - Ecole des Mines de Paris - 1A IC
# « Contribution au projet Open Source Substra », Owkin 

## Elèves

Jules Royer, Jules Désir, Benjamin Dougnac, Sam Pegeot

---

Langage: **Python**
Autre: *contribution à un projet open-source*
Librairie: **Tensorflow**

## Porteur du projet

Thibault Fouqueray [thibault.fouqueray@owkin.com](mailto:thibault.fouqueray@owkin.com), est ingénieur en machine-learning, et Romain Goussault (P09) [romain.goussault@owkin.com](mailto:romain.goussault@owkin.com) est Product Manager dans l'entreprise Owkin.

Ce projet vous propose de découvrir les bonnes pratiques de développement des projets open-source. Vous serez amenés à vous inspirer de codes existants pour développer vos programmes et à utiliser la librairie tensorflow. Ce projet peut déboucher sur une contribution open source. Il peut également vous permettre de vous familiariser avec de nouvelles techniques comme docker, kubernetes...

Pour ces raisons, ce projet s'adresse plutôr à un groupe d'élèves à l'aise avec la programmation et avec Python.

## Entreprise

Owkin [owkin.com](https://owkin.com/) est une entreprise (de 300 personnes) qui combine le machine-learning, l'IA et l'expertise dans le biomédical pour la résolution de problèmes médicaux. L'entreprise a un grand réseau de chercheurs (issus de centres universitaires), et des données de haute qualité sur des patients.

## Contexte du projet

Pour ses analyses, cette entreprise développe le projet open-source Substra ([github.com/Substra](https://github.com/Substra)) dont le but est de permettre des apprentissages fédérés c'est à dire des apprentissages qui s'effectuent sur un ensemble de bases de données (distantes) avec partage et synchronisation des modèles entraînés sur chacune d'elles.

Les données médicales de patients étant des données personnelles privées, pour des raisons de confidentialité et de sécurité, il n'est la plupart du temps pas possible de les regrouper pour les exploiter. D'où la nécessité de recourir à des apprentissages locaux, qui réalisent des modèles locaux, qui sont ensuite fédérés dans un modèle global.

## Description du projet

C'est dans le domaine du logiciel open source que ce projet prend place.

Substra ([docs.substra.org/en/stable](https://docs.substra.org/en/stable)) propose une API (Application Programming Interface) développée:

- pour PyTorch, un framework Python d'intelligence artificielle
- ainsi que pour Sickit-Learn un framework Python de machine-learning (voir [docs.substra.org/en/stable/substrafl_doc/examples/index.html](https://docs.substra.org/en/stable/substrafl_doc/examples/index.html)).

Mais Substra a pour vocation d'être compatible avec n'importe quel framework.

<aside>
💡 L'objectif de ce projet est d'implémenter un exemple d'utilisation du framework TensorFlow ([tensorflow.org](https://www.tensorflow.org/)) avec Substra. Le travail attendu devra avoir la forme d'une contribution open source et, le cas échéant, pourra être intégré à la documentation officielle du produit.

</aside>

Ce projet permettra aux élèves de découvrir et d'utiliser les méthodes, outils de développement et les bonnes pratiques qui *simplifient la vie aux programmeurs* et leur font *gagner du temps*, il leur fera aussi découvrir le framework (très utilisé) TensorFlow.

Le projet open source Substra est hébergé par github (https://github.com/Substra). Ce projet utilise différentes techniques mais les élèves resteront sur l'API 100% Python.

## Ressources mises à disposition des élèves

Les élèves auront accès à toute la stack du produit open source Substra.

## Avancement du travail effectué par le groupe

Nos classes Tensorflow sont enregistrées avec succès dans le `compute_plan`. Cependant les poids du modèle CNN ne sont pas mis à jour lors des phases d'agrégation, bien que le gestionnaire de poids `weight_manager` fonctionne lors des tests. 
