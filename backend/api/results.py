from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

class SignalParameters(BaseModel):
    carrierFrequency: float
    sampleRate: float
    bandwidth: float
    snr: float
    symbolRate: float

class HypothesisValidation(BaseModel):
    syncPassed: bool
    demodPassed: bool
    fecPassed: bool

class Hypothesis(BaseModel):
    id: str
    modulation: str
    symbolRate: float
    mlConfidence: float
    validation: HypothesisValidation
    isWinner: bool

class ResultsResponse(BaseModel):
    parameters: SignalParameters
    hypotheses: List[Hypothesis]

@router.get("/{analysis_id}", response_model=ResultsResponse)
async def get_results(analysis_id: str):
    parameters = SignalParameters(
        carrierFrequency=915e6,
        sampleRate=2.4e6,
        bandwidth=205e3,
        snr=18.6,
        symbolRate=200e3
    )
    
    hypotheses = [
        Hypothesis(
            id="hyp_1",
            modulation="QPSK",
            symbolRate=200e3,
            mlConfidence=0.942,
            validation=HypothesisValidation(
                syncPassed=True,
                demodPassed=True,
                fecPassed=True
            ),
            isWinner=True
        ),
        Hypothesis(
            id="hyp_2",
            modulation="BPSK",
            symbolRate=200e3,
            mlConfidence=0.038,
            validation=HypothesisValidation(
                syncPassed=True,
                demodPassed=True,
                fecPassed=False
            ),
            isWinner=False
        ),
        Hypothesis(
            id="hyp_3",
            modulation="8-PSK",
            symbolRate=200e3,
            mlConfidence=0.012,
            validation=HypothesisValidation(
                syncPassed=True,
                demodPassed=False,
                fecPassed=False
            ),
            isWinner=False
        ),
        Hypothesis(
            id="hyp_4",
            modulation="16-QAM",
            symbolRate=200e3,
            mlConfidence=0.008,
            validation=HypothesisValidation(
                syncPassed=False,
                demodPassed=False,
                fecPassed=False
            ),
            isWinner=False
        )
    ]
    
    return ResultsResponse(
        parameters=parameters,
        hypotheses=hypotheses
    )
