// SPDX-License-Identifier: Zlib
// Pattern algorithm adapted from vendored WDL/eel2/eel_strings.h. The only
// host substitutions are atomic guest references and the shared string store.
struct MatchWriter {
    std::string* text;
    const char* Get()const {return text->c_str();}
    int GetLength()const{return (int)text->size();}
    void Set(const char* s,int n=-1){if(n<0)*text=s;else text->assign(s,(size_t)n);}
    void SetRaw(const char* s,int n){text->assign(s,(size_t)n);}
};
struct MatchContext {
    Frame& frame; Strings& strings;
    int64_t formatHandle, messageHandle;
    std::deque<MatchWriter> writers;
    MatchWriter* writer(double handle) {
        const auto k=Strings::key(handle);
        // WDL protects the format and haystack from being overwritten by
        // captures, including captures reached during recursive backtracking.
        if(k==formatHandle || k==messageHandle || !strings.writable(handle))return nullptr;
        writers.push_back({&strings.values[k]});return &writers.back();
    }
    Reference namedVariable(const char* name,double* namedString) {
        std::string lower=name;
        for(char& c:lower)c=(char)std::tolower((unsigned char)c);
        if(!lower.empty() && lower[0]=='#') {
            const auto handle=frame.namedStringHandle(lower);
            if(handle){*namedString=(double)handle;return namedString;}
            return {};
        }
        if(auto* p=frame.cellPointer(lower.c_str()))return p;
        if(lower.rfind("slider",0)==0) {
            char* end=nullptr;const long index=std::strtol(lower.c_str()+6,&end,10);
            if(end && !*end && index>=1 && index<=DSPJSFX_MAX_SLIDERS)return &frame.scriptState().sliders[index-1];
        }
        return {};
    }
};
static int eel_string_match(MatchContext& context, const char *fmt, const char *msg, int match_fmt_pos, int ignorecase, const char *fmt_endptr, const char *msg_endptr, int num_fmt_parms, Reference* fmt_parms)
{
  // check for match, updating EEL_STRING_GETFMTVAR(*) as necessary
  // %d=12345
  // %f=12345[.678]
  // %c=any nonzero char, ascii value
  // %x=12354ab
  // %*, %?, %+, %% literals
  // * ? +  match minimal groups of 0+,1, or 1+ chars
  for (;;)
  {
    if (fmt>=fmt_endptr)
    {
      if (msg>=msg_endptr) return 1;
      return 0; // format ends before matching string
    }

    // if string ends and format is not on a wildcard, early-out to 0
    if (msg>=msg_endptr && *fmt != '*' && *fmt != '%') return 0;

    switch (*fmt)
    {
      case '*':
      case '+':
        // if last char of search pattern, we're done!
        if (fmt+1>=fmt_endptr || (fmt[1] == '?' && fmt+2>=fmt_endptr)) return *fmt == '*' || msg<msg_endptr;

        if (fmt[0] == '+')  msg++; // skip a character for + . Note that in this case msg[1] is valid, because of the !*msg && *fmt != '*' check above

        fmt++;
        if (*fmt == '?')
        {
          // *? or +? are lazy matches
          fmt++;

          while (msg<msg_endptr && !eel_string_match(context,fmt, msg,match_fmt_pos,ignorecase,fmt_endptr, msg_endptr,num_fmt_parms,fmt_parms)) msg++;
          return msg<msg_endptr;
        }
        else
        {
          // greedy match
          int len = (int) (msg_endptr-msg);
          while (len >= 0 && !eel_string_match(context,fmt, msg+len,match_fmt_pos,ignorecase,fmt_endptr, msg_endptr,num_fmt_parms,fmt_parms)) len--;
          return len >= 0;
        }
      break;
      case '?':
        fmt++;
        msg++;
      break;
      case '%':
        {
          fmt++;
          unsigned short fmt_minlen = 1, fmt_maxlen = 0;
          if (*fmt >= '0' && *fmt <= '9')
          {
            fmt_minlen = *fmt++ - '0';
            while (*fmt >= '0' && *fmt <= '9') fmt_minlen = fmt_minlen * 10 + (*fmt++ - '0');
            fmt_maxlen = fmt_minlen;
          }
          if (*fmt == '-')
          {
            fmt++;
            fmt_maxlen = 0;
            while (*fmt >= '0' && *fmt <= '9') fmt_maxlen = fmt_maxlen * 10 + (*fmt++ - '0');
          }
          const char *dest_varname=NULL;
          if (*fmt == '{')
          {
            dest_varname=++fmt;
            while (*fmt && fmt < fmt_endptr && *fmt != '}') fmt++;
            if (fmt >= fmt_endptr-1 || *fmt != '}') return 0; // malformed %{var}s
            fmt++; // skip '}'
          }

          char fmt_char = *fmt++;
          if (!fmt_char) return 0; // malformed

          if (fmt_char == '*' || 
              fmt_char == '?' || 
              fmt_char == '+' || 
              fmt_char == '%')
          {
            if (*msg++ != fmt_char) return 0;
          }
          else if (fmt_char == 'c')
          {
            Reference varOut;
            double vv=0.0;
            if (!dest_varname) 
            {
              if (match_fmt_pos < num_fmt_parms) varOut = fmt_parms[match_fmt_pos];
              match_fmt_pos++;
            }
            else
            {
 
              char tmp[128];
              int idx=0;
              while (dest_varname < fmt_endptr && *dest_varname && *dest_varname != '}' && idx<(int)sizeof(tmp)-1) tmp[idx++] = *dest_varname++;
              tmp[idx]=0;
              if (idx>0) varOut = context.namedVariable(tmp,&vv);

            }
            if (msg >= msg_endptr) return 0; // out of chars

            if (varOut) 
            {
              if (varOut == &vv) // %{#foo}c
              {
                MatchWriter *wr=NULL;
                wr=context.writer(vv);
                if (wr) wr->Set(msg,1);
              }
              else
              {
                *varOut = (double)*(unsigned char *)msg;
              }
            }
            msg++;
          }
          else 
          {
            int len=0;
            int lazy=0;
            if (fmt_char>='A'&&fmt_char<='Z') { lazy=1; fmt_char += 'a' - 'A'; }

            if (fmt_char == 's')
            {
              len = (int) (msg_endptr-msg);
            }
            else if (fmt_char == 'x')
            {
              while ((msg[len] >= '0' && msg[len] <= '9') ||
                     (msg[len] >= 'A' && msg[len] <= 'F') ||
                     (msg[len] >= 'a' && msg[len] <= 'f')) len++;
            }
            else if (fmt_char == 'f')
            {
              if (msg[len] == '-') len++;
              while (msg[len] >= '0' && msg[len] <= '9') len++;
              if (msg[len] == '.') 
              { 
                len++; 
                while (msg[len] >= '0' && msg[len] <= '9') len++;
              }
            }
            else if (fmt_char == 'd' || fmt_char == 'u' || fmt_char == 'i')
            {
              if (fmt_char != 'u' && msg[len] == '-') len++;
              while (msg[len] >= '0' && msg[len] <= '9') len++;
            }
            else 
            {
              // bad format
              return 0;
            }

            if (fmt_maxlen>0 && len > fmt_maxlen) len = fmt_maxlen;

            if (!dest_varname) match_fmt_pos++;

            if (lazy)
            {
              if (fmt_maxlen<1 || fmt_maxlen>len) fmt_maxlen=len;
              len=fmt_minlen;
              while (len <= fmt_maxlen && !eel_string_match(context,fmt, msg+len,match_fmt_pos,ignorecase,fmt_endptr, msg_endptr,num_fmt_parms,fmt_parms)) len++;
              if (len > fmt_maxlen) return 0;
            }
            else
            {
              while (len >= fmt_minlen && !eel_string_match(context,fmt, msg+len,match_fmt_pos,ignorecase,fmt_endptr, msg_endptr,num_fmt_parms,fmt_parms)) len--;
              if (len < fmt_minlen) return 0;
            }

            double vv=0.0;
            Reference varOut;
            if (!dest_varname) 
            {
              if (match_fmt_pos>0 && match_fmt_pos-1 < num_fmt_parms) varOut = fmt_parms[match_fmt_pos-1];
            }
            else
            {
 
              char tmp[128];
              int idx=0;
              while (dest_varname < fmt_endptr && *dest_varname  && *dest_varname != '}' && idx<(int)sizeof(tmp)-1) tmp[idx++] = *dest_varname++;
              tmp[idx]=0;
              if (idx>0) varOut = context.namedVariable(tmp,&vv);

            }
            if (varOut)
            {
              if (fmt_char == 's')
              {
                MatchWriter *wr=NULL;
                wr=context.writer(*varOut);
                if (wr)
                {
                  if (msg_endptr >= wr->Get() && msg_endptr <= wr->Get() + wr->GetLength())
                  {
                  }
                  else if (fmt_endptr >= wr->Get() && fmt_endptr <= wr->Get() + wr->GetLength())
                  {
                  }
                  else 
                  {
                    wr->SetRaw(msg,len);
                  }
                }
                else
                {
                }
              }
              else
              {
                char tmp[128];
                lstrcpyn_safe(tmp,msg,wdl_min(len+1,(int)sizeof(tmp)));
                if (varOut == &vv) 
                {
                  MatchWriter *wr=NULL;
                  wr=context.writer(vv);
                  if (wr) wr->Set(tmp);
                }
                else
                {
                  char *bl=(char*)msg;
                  if (fmt_char == 'u')
                    *varOut = (double)strtoul(tmp,&bl,10);
                  else if (fmt_char == 'x')
                    *varOut = (double)strtoul(msg,&bl,16);
                  else
                    *varOut = (double)atof(tmp);
                }
              }
            }
            return 1;
          }
        }
      break;
      default:
        if (ignorecase ? (toupper(*fmt) != toupper(*msg)) : (*fmt!= *msg)) return 0;
        fmt++;
        msg++;
      break;
    }
  }
}




double matchStrings(int count,Reference* args,bool ignorecase) {
    if(count<2)return 0;
    auto* strings=stringContext(&state);if(!strings)return 0;
    std::lock_guard lock(strings->mutex);
    const double fmtHandle=*args[0],msgHandle=*args[1];
    const auto format=strings->readUnlocked(fmtHandle), message=strings->readUnlocked(msgHandle);
    MatchContext context{*this,*strings,Strings::key(fmtHandle),Strings::key(msgHandle),{}};
    return eel_string_match(context,format.c_str(),message.c_str(),0,ignorecase,
        format.data()+format.size(),message.data()+message.size(),count-2,args+2);
}
