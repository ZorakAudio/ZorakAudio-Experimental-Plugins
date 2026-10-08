// SPDX-License-Identifier: Zlib
#pragma once
#include <cmath>
#include <optional>
#include <string>
#include <vector>
namespace za::jsfx {
// REAPER reserves file handle zero for the current @serialize transaction.
// The caller owns the save/restore boundary; normal file handles are unaffected.
struct Serialization {
    enum Operation {Avail,Var,Mem,String};
    std::vector<double> cells;
    size_t cursor=0;
    bool reading=false,active=false;
    std::optional<double> dispatch(DSPJSFX_State& state,Operation op,double handle,double* value=nullptr,double base=0,double length=0) {
        if(!active || handle!=0)return {};
        if(op==Avail)return reading?double(cells.size()-cursor):-1.0;
        if(op==Var) {
            if(!value)return 0;
            if(reading){if(cursor>=cells.size())return 0;jsfxCellStore(value,cells[cursor++]);}
            else cells.push_back(jsfxCellLoad(value));
            return 1;
        }
        if(op==Mem) {
            if(!std::isfinite(base)||!std::isfinite(length)||base<0||length<0||base>state.memN)return 0;
            auto first=(int64_t)base;auto count=(int64_t)std::min(length,double(state.memN-first));
            int64_t copied=0;
            for(;copied<count;++copied){if(reading){if(cursor>=cells.size())break;state.mem[first+copied]=cells[cursor++];}else cells.push_back(double(state.mem[first+copied]));}
            return double(copied);
        }
        if(!value)return 0;
        if(reading) {
            if(cursor>=cells.size())return 0;
            double raw=cells[cursor++];
            if(!std::isfinite(raw)||raw<0||raw>16383||raw>cells.size()-cursor)return 0;
            std::string text;for(size_t n=0;n<(size_t)raw;++n)text.push_back((char)(unsigned char)cells[cursor++]);
            return jsfx_string_assign_utf8(&state,value,text.data(),(int)text.size())?1.0:0.0;
        }
        auto handleValue=jsfxCellLoad(value);int count=std::clamp((int)jsfx_strlen(&state,handleValue),0,16383);
        cells.push_back(double(count));for(int i=0;i<count;++i)cells.push_back(jsfx_str_getchar(&state,handleValue,double(i)));
        return 1;
    }
};
}
