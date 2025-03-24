%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Coupled class for LaTeX files                 v1.1 Feb-23-2000 %%
%%                                                                %%
%% The regular article class commands still exist and may be used.%%
%% 10pt, 11pt and twocolumn options are disabled.                 %%
%%                                                                %%
%% \maketitle make the first page automatically                   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
%\ProvidesClass{coupled}[2009/06/8-11 COUPLED paper class]

\DeclareOption{10pt}{\OptionNotUsed}
\DeclareOption{11pt}{\OptionNotUsed}
\DeclareOption{twocolumn}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

\LoadClass[11pt]{article}

\RequirePackage{calc}
\RequirePackage{indentfirst}

%% Title 
\renewcommand{\maketitle}{%
 \newpage
 \thispagestyle{plain}
 
 \begin{minipage}[t][1cm][t]{16.5cm}
    \begin{flushright}
        \fontsize{10}{10}\selectfont
        \vspace{-50pt}
        \textit{Preprint submitted to MARINE 2025}
    \end{flushright}
 \end{minipage}
 
 \begin{center}%
  \vspace{-60pt}%
  {%3
   \fontsize{14}{14}\selectfont
   \bfseries
    {\@title}
   \par
  }%3
  %\vspace{18pt}%
 \vspace{10pt}%
  {%4
\fontsize{12}{12}\selectfont
   \bfseries\@author
   \par
  }%4
  %\vspace{14pt}%
  \vspace{8pt} % Space between authors and affiliation
  {%5
\fontsize{11}{12}\selectfont
   %\def\and{\\\vspace{14pt}}
\def\and{\\\vspace{13pt}} % Space between affiliation and corresponding author
   \normalfont\@address
   \par
  }%5
 \end{center}
 %\vspace{18pt}%

 \vspace{8pt} %
\centerline{\textbf{ABSTRACT}}{\textrm\@abstract}
 
 % \noindent\textbf{Keywords:\space}{\textrm\@abstract}
 \par

  \vspace{11pt}%
 \noindent\textbf{Keywords:\space}\@keywords
 \par
 
%\vspace{6pt}%
 %\vfill
 %\newpage
}% End of \@maketitle

\def\author{\@ifnextchar[{\@getshortauthor}{\@getauthor}}
\def\@getauthor#1{\gdef\@author{#1}\gdef\@shortauthor{#1}}
\def\@getshortauthor[#1]#2{\gdef\@shortauthor{#1}\gdef\@author{#2}}

\newcommand{\address}[1]{\gdef\@address{#1}}

\newcommand{\keywords}[1]{\gdef\@keywords{#1}}

\renewcommand{\abstract}[1]{\gdef\@abstract{#1}}

% Header settings
\newcommand{\heading}[1]{\gdef\@heading{#1}}

% This removes the heading with author names, keeps the numbers at foot in the center
\def\ps@myheadings{%
    \def\@oddfoot{\normalfont\hfil\thepage\hfil} % Keep page number centered
    \let\@evenfoot\@oddfoot % Ensure even pages match
    \def\@evenhead{} % Remove header
    \def\@oddhead{} % Remove header for odd pages as well
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
}

\def\sectionLevel{1}
\def\sectionIndent{0pt}
\def\sectionSpaceBefore{12pt}
\def\sectionSpaceAfter{6pt}
\def\sectionStyle{\normalsize\bfseries\MakeUppercase}

\renewcommand{\section}{%
  \@startsection{section}{\sectionLevel}{\sectionIndent}{\sectionSpaceBefore}%
                {\sectionSpaceAfter}{\sectionStyle}}%


\def\sectionLevel{2}
\def\sectionIndent{0pt}
\def\sectionSpaceBefore{12pt}
\def\sectionSpaceAfter{6pt}
\def\sectionStyle{\normalsize\bfseries}

\renewcommand{\subsection}{%
  \@startsection{subsection}{\sectionLevel}{\sectionIndent}{\sectionSpaceBefore}%
                {\sectionSpaceAfter}{\sectionStyle}}%


\renewcommand{\@makecaption}[2]{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\normalsize\fontsize{10}{12}\selectfont {\bf #1}: #2}%
  \ifdim \wd\@tempboxa >\hsize
    \normalsize\fontsize{10}{12}\selectfont {\bf #1}: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

\renewenvironment{thebibliography}[1]
  {%\newpage
   \section*{\MakeUppercase\refname\@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
   \list{\@biblabel{\@arabic\c@enumiv}}%
        {\settowidth\labelwidth{\@biblabel{#1}}%
         \leftmargin\labelwidth
         \advance\leftmargin\labelsep
         \@openbib@code
         \usecounter{enumiv}%
         \let\p@enumiv\@empty
         \renewcommand\theenumiv{\@arabic\c@enumiv}}%
   \sloppy
   \clubpenalty4000
   \@clubpenalty \clubpenalty
   \widowpenalty4000%
   \sfcode`\.\@m}
  {\def\@noitemerr
   {\@latex@warning{Empty `thebibliography' environment}}%
   \endlist}

% MARINE
\setlength{\topmargin}{0cm}
\setlength{\oddsidemargin}{0mm}
\setlength{\textheight}{230mm}
\setlength{\textwidth}{170mm}
\setlength{\textfloatsep}{12pt}
\setlength{\abovedisplayskip}{6pt}
\setlength{\belowdisplayskip}{6pt}
\setlength{\parindent}{0mm}
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{6pt}
\setlength{\itemsep}{0pt}
\setlength{\voffset}{-1.7cm}
\setlength{\footskip}{1cm}
\AtBeginDocument{%
  \pagestyle{myheadings}
  \maketitle
  \let\maketitle\relax
}

\endinput
