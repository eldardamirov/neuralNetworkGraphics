    //
//  main.cpp
//  neuralNetwork
//
//  Created by Эльдар Дамиров on 18.08.17.
//  Copyright © 2017 Эльдар Дамиров. All rights reserved.
//
#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include "ResourcePath.hpp"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <random>
#include <assert.h>
#include <cmath>
#include <sstream>

const int screenX = 1300;
const int screenY = 480;



void printVector ( std::string label, std::vector <double> &myVector );
void printGraph ( std::vector <double> errorMemory, sf::RenderWindow& window );



class trainingData
    {
    public:
        trainingData ( const std::string filename );

        bool isEOF ( void )
            {
            return ( trainingDataFile.eof() );
            }

        void getTopology ( std::vector <unsigned> &topology );

        unsigned getNextInputs ( std::vector <double> &inputValues );
        unsigned getTargetOutputs ( std::vector <double> &targetOutputValues );


    private:
        std::ifstream trainingDataFile;




    };








class neuron;


typedef std::vector <neuron> layer;

struct connection
    {
    double weight;
    double deltaWeight;
    };


class neuron
    {
    public:
    
        std::vector <connection> outputWeights;

        neuron ( unsigned numberOfOutputs, unsigned myIndexTemp, std::vector <double> weight, std::vector <double> deltaWeight );
        void feedForward ( const layer &previousLayer );

        double getOutputValue() const
            {
            return outputValue;
            }

        void setOutputValue ( double newValue )
            {
            outputValue = newValue;
            }

        void calculateOutputGradients ( double targetValue );
        void calculateHiddenGradients ( layer& nextLayer );
        void updateInputWeights ( layer& previousLayer );
        


    private:
        /*
        %eta% --    0.0 - slow learning
                    0.2 - medium learning
                    1.0 - reckless learning
        
        
        %alpha% --  0.0 - no momentum   
                    0.5 - moderate momentum
        
        */

        static double eta; //  0.0 ... 1.0    //////////   !!!!!!!
        static double alpha; // 0.0 ...... INFINITY  //////////   !!!!!!!


        double outputValue;
        unsigned myIndex;
        double gradient;
        
        double randomWeight ()
            {
            return ( rand() / double ( RAND_MAX ) );
            }


        static double transferFunction ( double x );
        static double transferFunctionDerivative ( double x );
        double sumOfDerivatives ( const layer& nextLayer ) const;


    };


double neuron::eta = 0.7;  ////////////////////////////////////  !!!!!  TUNEABLE  !!!!! ////////////////////////////////////
double neuron::alpha = 0.35; ////////////////////////////////////  !!!!!  TUNEABLE  !!!!! ////////////////////////////////////




class net
    {
    public:
        net ( const std::vector <unsigned> &topology );

        void feedForward ( const std::vector <double> &inputValues );
        void backPropogation ( const std::vector <double> &targetValues );  // learning;
        void getResults ( std::vector <double> &resultValues ) const;
        std::vector <layer> layers;
        
        double getRecentAverageError ( void ) const
            {
            return recentAverageError;
            }


    private:
        double error;
        double recentAverageError;
        double recentAverageSmoothingFactor;


    };


int main()
    {
    sf::RenderWindow window ( sf::VideoMode ( screenX, screenY ), "Graph" );
    
    trainingData trainData ( "trainData.txt" );
    std::vector <unsigned> topology;
    trainData.getTopology ( topology );
    net myNet ( topology );
    std::ofstream file("NeuronWeights.txt");



    std::vector <double> inputValues;
    std::vector <double> targetValues;
    std::vector <double> resultValues;
    
    std::vector <double> errorMemory;

    int trainingPass = 0;
    
    bool isPrint = false;
    
    while ( !trainData.isEOF() )
        {
        trainingPass++;
            
        if ( trainData.getNextInputs ( inputValues ) != topology [ 0 ] )
            {
            break;
            }
            
            
        myNet.feedForward ( inputValues );

        myNet.getResults ( resultValues );
        

        trainData.getTargetOutputs ( targetValues );
        if ( isPrint )
            {
            std::cout << std::endl << "Pass " << trainingPass;
            
            printVector ( ": Input:", inputValues );
            printVector ( "Outputs: ", resultValues );
            printVector ( "Targets: " , targetValues );
            std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
            }


        assert ( targetValues.size() == topology.back() ); /////
        
        errorMemory.push_back ( std::abs ( ( resultValues [ 0 ] - targetValues [ 0 ] * 100 ) / targetValues [ 0 ] ) );

        myNet.backPropogation ( targetValues );

        }


    
    
    
    for ( int currentLayer = 0; currentLayer < myNet.layers.size(); currentLayer++ )
        {
        for ( int currentNeuron = 0; currentNeuron < myNet.layers [ currentLayer ].size(); currentNeuron++ )
            {
            for ( int currentOutputWeight = 0; currentOutputWeight < myNet.layers [ currentLayer ] [ currentNeuron ].outputWeights.size(); currentOutputWeight++ )
                {
                file << myNet.layers [ currentLayer ] [ currentNeuron ].outputWeights [ currentOutputWeight ].weight << " " << myNet.layers [ currentLayer ] [ currentNeuron ].outputWeights [ currentOutputWeight ].deltaWeight << std::endl;
                }
            }
        }

    file.close();

    printGraph ( errorMemory, window );
    
    while ( window.isOpen() )
        {
        sf::Event event;
        while ( window.pollEvent ( event ) )
            {
            if ( event.type == sf::Event::Closed ) window.close();
            }
        }


    return 0;
    }








net::net ( const std::vector <unsigned> &topology )
    {
    unsigned long numberOfLayers = topology.size();
    
    std::ifstream neuronWeights ( "neuronWeights.txt" );

    for ( unsigned currentLayer = 0; currentLayer < numberOfLayers; currentLayer++ )
        {
        layers.push_back ( layer() );
        unsigned numberOfOutputs = currentLayer == ( topology.size() - 1 ) ? 0 : topology [ currentLayer + 1 ];

        for ( unsigned currentNeuronInLayer = 0; currentNeuronInLayer <= topology [ currentLayer ]; currentNeuronInLayer++ ) // <= because of bias neuron;
            {
            std::vector <double> weight, deltaWeight; 
            for ( unsigned output = 0; output < numberOfOutputs; output++ )
                {
                double temp = 0.0;
                neuronWeights >> temp;
                weight.push_back ( temp );
                                
                neuronWeights >> temp;
                deltaWeight.push_back ( temp );
                }
            layers.back().push_back ( neuron ( numberOfOutputs, currentNeuronInLayer, weight, deltaWeight ) );
            //printf ( "Hello I'm new neuron in layer: %d, my index is: %d.\n", currentLayer, currentNeuronInLayer );
            }

        layers.back().back().setOutputValue ( 0.0 );
        }


    }



void net::feedForward ( const std::vector<double> &inputValues )
    {
    assert ( inputValues.size() == ( layers [ 0 ].size() - 1 ) );

    for ( unsigned input = 0; input < inputValues.size(); input++ )
        {
        layers [ 0 ] [ input ].setOutputValue ( inputValues [ input ] );
        }


    for ( unsigned currentLayer = 1; currentLayer < layers.size(); currentLayer++ )
        {
        layer &previousLayer = layers [ currentLayer - 1 ];
        for ( unsigned currentNeuron = 0; currentNeuron < layers [ currentLayer ].size() - 1; currentNeuron++ )
            {
            layers [ currentLayer ] [ currentNeuron ].feedForward ( previousLayer );
            }
        }
    }


void net::backPropogation ( const std::vector<double> &targetValues )
    {
    layer &outputLayer = layers.back(); // all error in all net;

    double sumError = 0.0;
    double delta = 0.0;
    for ( unsigned currentNeuron = 0; currentNeuron < ( outputLayer.size() - 1 ); currentNeuron++ )
        {
        delta = targetValues [ currentNeuron ] - outputLayer [ currentNeuron ].getOutputValue();
        sumError = sumError + ( delta * delta );
        }

    sumError = sumError / ( outputLayer.size() - 1 );
    sumError = sqrt ( sumError );

    ////

    recentAverageError = ( recentAverageError * recentAverageSmoothingFactor + sumError ) / ( recentAverageSmoothingFactor + 1.0 );

    if ( !isfinite ( recentAverageError ) )
        {
        abort();
        }

    ////

    for ( unsigned currentNeuron = 0; currentNeuron < ( outputLayer.size() - 1 ); currentNeuron++ )
        {
        outputLayer [ currentNeuron ].calculateOutputGradients ( targetValues [ currentNeuron ] );
        }


    for ( long long currentLayer = ( layers.size() - 2 ); currentLayer > 0; currentLayer-- )
        {
        layer &currentHiddenLayer = layers [ currentLayer ];
        layer &nextHiddenLayer = layers [ currentLayer + 1 ];

        for ( unsigned currentNeuron = 0; currentNeuron < currentHiddenLayer.size(); currentNeuron++ )
            {
            currentHiddenLayer [ currentNeuron ].calculateHiddenGradients ( nextHiddenLayer );
            }

        }

    ////

    for ( long long currentLayerNumber = ( layers.size() - 1 ); currentLayerNumber > 0; currentLayerNumber-- )
        {
        layer &previousLayer = layers [ currentLayerNumber - 1 ];
        layer &currentLayer = layers [ currentLayerNumber ];

        for ( int currentNeuron = 0; currentNeuron < ( currentLayer.size() - 1 ); currentNeuron++ )
            {
            currentLayer [ currentNeuron ].updateInputWeights ( previousLayer );
            }

        }


    }


void net::getResults ( std::vector<double> &resultValues ) const
    {
    resultValues.clear();

    for ( unsigned currentNeuron = 0; currentNeuron < ( layers.back().size() - 1 ); currentNeuron++ )
        {
        resultValues.push_back ( layers.back() [ currentNeuron ].getOutputValue() );
        }
    }



////////




neuron::neuron ( unsigned numberOfOutput, unsigned myIndexTemp, std::vector <double> weight, std::vector <double> deltaWeight )
    {
    for ( unsigned connections = 0; connections < numberOfOutput; connections++ )
        {
        outputWeights.push_back ( connection() );
        //outputWeights.back().weight = randomWeight();
        if ( weight [ connections ] != 0 ) 
            {
            outputWeights.back().weight = weight [ connections ];
            }
        else
            {
            outputWeights.back().weight = randomWeight();
            }
            
        if ( deltaWeight [ connections ] != 0 ) 
            {
            outputWeights.back().deltaWeight = deltaWeight [ connections ];
            }
        else
            {
            outputWeights.back().deltaWeight = randomWeight();
            }
        
        }


    myIndex = myIndexTemp;

    printf ( "Hello I'm new neuron, my index is: %d ( %d ) and I have %d outputs.\n", myIndex , myIndex, numberOfOutput );
    //myIndex = myIndex;
    //myIndex = myIndex;
    //myIndex = myIndex;


    }

void neuron::feedForward ( const layer &previousLayer )
    {
    double sum = 0.0;

    for ( unsigned previousLayerNeuron = 0; previousLayerNeuron < previousLayer.size(); previousLayerNeuron++ )
        {
        sum = sum + previousLayer [ previousLayerNeuron ].getOutputValue() * previousLayer [ previousLayerNeuron ].outputWeights [ myIndex ].weight;
        }

    outputValue = neuron::transferFunction ( sum );
    }


double neuron::transferFunction ( double x )
    {
    // [ - 1.0 ... 1.0 ];

    return tanh ( x );
    //return sin ( x );
    }

double neuron::transferFunctionDerivative ( double x )
    {
    return ( 1.0 - ( x * x ) );
    //return cos ( x );
    }


void neuron::updateInputWeights ( layer &previousLayer )
    {
    double previousDeltaWeight = 0.0;
    double newDeltaWeight = 0.0;



    for ( unsigned currentNeuron = 0; currentNeuron < previousLayer.size(); currentNeuron++ )
        {
        neuron &neuron = previousLayer [ currentNeuron ];

        previousDeltaWeight = neuron.outputWeights [ myIndex ].deltaWeight;

        newDeltaWeight = ( eta * neuron.getOutputValue() * gradient ) + ( alpha * previousDeltaWeight );


        neuron.outputWeights [ myIndex ].deltaWeight = newDeltaWeight;
        neuron.outputWeights [ myIndex ].weight = neuron.outputWeights [ myIndex ].weight + newDeltaWeight;
        }

    }


double neuron::sumOfDerivatives ( const layer& nextLayer ) const
    {
    double sum = 0.0;

    for ( unsigned currentNeuron = 0; currentNeuron < ( nextLayer.size() - 1 ); currentNeuron++ )
        {
        sum = sum + outputWeights [ currentNeuron ].weight * nextLayer [ currentNeuron ].gradient;
        }

    return sum;
    }


void neuron::calculateOutputGradients ( double targetValue )
    {
    double delta = targetValue - outputValue;
    gradient = delta * neuron::transferFunctionDerivative ( outputValue );
    }

void neuron::calculateHiddenGradients ( layer& nextLayer )
    {
    double derivativeWeigths = sumOfDerivatives ( nextLayer );
    gradient = derivativeWeigths * neuron::transferFunctionDerivative ( outputValue );
    }






trainingData::trainingData ( const std::string filename )
    {
    trainingDataFile.open ( filename.c_str() );
    }





void trainingData::getTopology ( std::vector<unsigned int> &topology )
    {
    std::string line = "";
    std::string label = "";

    getline ( trainingDataFile, line );

    std::stringstream ss ( line );
    ss >> label;

    if ( ( isEOF() ) || ( label.compare ( "topology:" ) != 0 ) )
        {
        std::cout << label << "Topology missing." << std::endl;
        abort();
        }

    unsigned temp = 0;

    while ( !ss.eof() )
        {
        ss >> temp;
        topology.push_back ( temp );
        }

    }




unsigned trainingData::getNextInputs ( std::vector <double> &inputValues )
    {
    inputValues.clear();

    std::string line;
    getline( trainingDataFile, line );

    std::stringstream ss ( line );

    std::string label;
    ss >> label;
    double temp = 0.0;

    while ( !ss.eof() )
        {
        while ( ss >> temp )
            {
            inputValues.push_back ( temp );
            }
        }

    return inputValues.size();
    }


unsigned trainingData::getTargetOutputs ( std::vector <double> &targetOutputValues )
    {
    targetOutputValues.clear();

    std::string line;
    getline ( trainingDataFile, line );
    std::stringstream ss ( line );

    std::string label;
    ss >> label;

    double temp = 0.0;

    if ( label.compare ( "out:" ) == 0 )
        {
        while ( ss >> temp )
            {
            targetOutputValues.push_back ( temp );
            }
        }

    return targetOutputValues.size();
    }




void printVector ( std::string label, std::vector <double> &myVector )
    {
    printf ( "%s ", label.c_str() );

    for ( unsigned i = 0; i < myVector.size(); i++ )
        {
        //std::cout << ( myVector [ i ] ) << " ";
        std::cout << ( 1 / myVector [ i ] ) << " ";
        }
    printf ( "\n" );
    }



void printGraph ( std::vector <double> errorMemory, sf::RenderWindow& window )
    {
    sf::VertexArray lines ( sf::LinesStrip, errorMemory.size() );
    
    double maximum = 0, middle = 0;
    double step = ( screenX - 40 ) / double ( errorMemory.size() );
    
    //std::cout << step << std::endl;
    
    for ( int i = 0; i < errorMemory.size(); i++ )
        {
        middle += errorMemory [ i ] / errorMemory.size();
        maximum = std::max ( maximum, errorMemory [ i ] );
        }
        
    double coefficent = ( screenY - 40 ) / maximum;
    
    for ( int i = 0; i < errorMemory.size(); i++ )
        {
        lines [ i ].position = sf::Vector2f ( 20 + ( step * i ), ( 20 + screenY - ( coefficent * errorMemory [ i ] ) ) );
        lines [ i ].color = sf::Color ( ( ( 255 / maximum ) * errorMemory [ i ] ), ( 255 - ( ( 255 / maximum ) * errorMemory [ i ] ) ) , 0 );
        //lines [ i ].color = sf::Color::White;
        }
        
    
        
    window.clear();
    window.draw ( lines );
    window.display();
    }
