import {apiGet} from "./client.js";

export function fetchOutcomes() {
    return apiGet("/outcomes");
}


export function fetchSchema(outcomeId){
    return apiGet(`/schema/${outcomeId}`);
}

