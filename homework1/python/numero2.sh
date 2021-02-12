#!/bin/bash

echo "Making mocks"
python numero2.py --create_mocks 
echo "Count in spheres"
python numero2.py --estimator=count-in-sphere
python numero2.py --plot_results --estimator=count-in-sphere
echo "Hamilton"
python numero2.py --estimator=Hamilton
python numero2.py --plot_results --estimator=Hamilton
echo "Peebles Davis"
python numero2.py --estimator=Peebles-Davis
python numero2.py --plot_results --estimator=Peebles-Davis
echo "Landy-Svaley"
python numero2.py --estimator=Landy-Szalay
python numero2.py --plot_results --estimator=Landy-Szalay
